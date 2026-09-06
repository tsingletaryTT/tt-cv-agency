#!/usr/bin/env python3
"""Round-trip proof: send OSC /fader values into VCV Rack (via OSC'elot),
capture VCV's audio output from the vcv_loop loopback sink, and confirm
the two are correlated. Also listens for OSC'elot's feedback messages
(SEND port) to confirm the parameter value round-trips back out.
"""
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

OSCELOT_RECEIVE_PORT = 8881  # send our control messages here
OSCELOT_SEND_PORT = 8880     # OSC'elot sends feedback here
LOOPBACK_MONITOR = "vcv_loop.monitor"
BLOCK_SIZE = 1024
SAMPLE_RATE = 48000

feedback_events = []  # (wall_clock_time, address, args)


def feedback_handler(address, *args):
    feedback_events.append((time.monotonic(), address, args))


def start_feedback_listener():
    disp = Dispatcher()
    disp.map("/fader", feedback_handler)
    disp.map("/*/info", feedback_handler)
    disp.set_default_handler(feedback_handler)
    server = BlockingOSCUDPServer(("127.0.0.1", OSCELOT_SEND_PORT), disp)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def find_loopback_device():
    # PortAudio's pulse host API doesn't expose named Pulse sinks/sources --
    # it only ever shows a generic "pulse" pseudo-device. Targeting the
    # right physical source is done by setting the system default source
    # to vcv_loop.monitor before running (see README), and using that
    # generic device here.
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["name"] == "pulse" and d["max_input_channels"] > 0:
            return i
    raise RuntimeError("Could not find the 'pulse' PortAudio device")


def main():
    print("Starting OSC feedback listener on port", OSCELOT_SEND_PORT)
    start_feedback_listener()

    dev_index = find_loopback_device()
    print("Using audio input device:", sd.query_devices(dev_index)["name"])

    rms_log = []  # (wall_clock_time, rms)
    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        rms = float(np.sqrt(np.mean(indata.astype(np.float64) ** 2)))
        audio_q.put((time.monotonic(), rms))

    stream = sd.InputStream(
        device=dev_index,
        channels=2,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    )

    client = SimpleUDPClient("127.0.0.1", OSCELOT_RECEIVE_PORT)

    with stream:
        # Drain a moment of baseline audio first.
        time.sleep(0.5)
        while not audio_q.empty():
            audio_q.get()

        sweep_values = [0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0]
        for value in sweep_values:
            send_time = time.monotonic()
            client.send_message("/fader", [1, value])
            print(f"--> sent /fader 1 {value:.2f} at t={send_time:.4f}")
            time.sleep(1.0)  # let audio settle and log a window of RMS

            # Drain whatever RMS samples arrived in this window.
            window = []
            while not audio_q.empty():
                window.append(audio_q.get())
            if window:
                avg_rms = sum(r for _, r in window) / len(window)
                first_t = window[0][0]
                latency_ms = (first_t - send_time) * 1000
                print(
                    f"    observed {len(window)} audio blocks, "
                    f"avg RMS={avg_rms:.5f}, first-block latency={latency_ms:.1f}ms"
                )
            else:
                print("    WARNING: no audio blocks captured in this window")

    print("\nFeedback messages received from OSC'elot:")
    for t, addr, args in feedback_events:
        print(f"  t={t:.4f} {addr} {args}")
    if not feedback_events:
        print("  (none received -- check that OSC'elot's SEND toggle is on)")


if __name__ == "__main__":
    main()
