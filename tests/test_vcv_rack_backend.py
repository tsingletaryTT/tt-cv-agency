# tests/test_vcv_rack_backend.py
import queue
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from backends.vcv_rack import AudioBlockTimeoutError, VCVRackBackend, load_channel_config


def test_load_channel_config_reads_cc_numbers(tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text(
        "midi_port_name: 'Fake Port'\n"
        "channels:\n"
        "  chan_a:\n"
        "    cc: 5\n"
        "  chan_b:\n"
        "    cc: 9\n"
    )
    midi_port_name, channel_cc = load_channel_config(str(config_path))
    assert midi_port_name == "Fake Port"
    assert channel_cc == {"chan_a": 5, "chan_b": 9}


@patch("backends.vcv_rack.sd.InputStream")
@patch("backends.vcv_rack.mido.open_output")
def test_set_cv_sends_control_change_with_scaled_value(mock_open_output, mock_input_stream, tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text(
        "midi_port_name: 'Fake Port'\nchannels:\n  vca_level:\n    cc: 3\n"
    )
    mock_port = MagicMock()
    mock_open_output.return_value = mock_port

    backend = VCVRackBackend(config_path=str(config_path))
    backend.set_cv("vca_level", 0.5)

    assert mock_port.send.call_count == 1
    sent_msg = mock_port.send.call_args[0][0]
    assert sent_msg.type == "control_change"
    assert sent_msg.control == 3
    assert sent_msg.value == round(0.5 * 127)


@patch("backends.vcv_rack.sd.InputStream")
@patch("backends.vcv_rack.mido.open_output")
def test_set_cv_rejects_unknown_channel(mock_open_output, mock_input_stream, tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text("midi_port_name: 'Fake Port'\nchannels:\n  a:\n    cc: 1\n")
    backend = VCVRackBackend(config_path=str(config_path))
    with pytest.raises(KeyError):
        backend.set_cv("nonexistent", 0.5)


@patch("backends.vcv_rack.sd.InputStream")
@patch("backends.vcv_rack.mido.open_output")
def test_channels_returns_config_keys(mock_open_output, mock_input_stream, tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text(
        "midi_port_name: 'Fake Port'\nchannels:\n  a:\n    cc: 1\n  b:\n    cc: 2\n"
    )
    backend = VCVRackBackend(config_path=str(config_path))
    assert backend.channels() == ["a", "b"]


@patch("backends.vcv_rack.sd.InputStream")
@patch("backends.vcv_rack.mido.open_output")
def test_sample_rate_and_block_size_report_real_configured_values(mock_open_output, mock_input_stream, tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text("midi_port_name: 'Fake Port'\nchannels:\n  a:\n    cc: 1\n")
    # Production defaults (48000 Hz / 1024 samples) unless overridden.
    backend = VCVRackBackend(config_path=str(config_path))
    assert backend.sample_rate() == 48000
    assert backend.block_size() == 1024

    backend2 = VCVRackBackend(config_path=str(config_path), sample_rate=44100, block_size=256)
    assert backend2.sample_rate() == 44100
    assert backend2.block_size() == 256


@patch("backends.vcv_rack.sd.InputStream")
@patch("backends.vcv_rack.mido.open_output")
def test_read_audio_block_raises_clear_timeout_error_when_stream_stalls(mock_open_output, mock_input_stream, tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text("midi_port_name: 'Fake Port'\nchannels:\n  a:\n    cc: 1\n")
    backend = VCVRackBackend(config_path=str(config_path), read_timeout_s=0.05)

    # Simulate a stalled audio source: the queue never receives a block, so
    # a plain `.get()` with no timeout would block forever. Replace the
    # real (empty) queue with a mock whose .get() raises queue.Empty
    # immediately, so the test doesn't actually have to wait out the
    # timeout to exercise the failure path.
    backend._audio_queue = MagicMock()
    backend._audio_queue.get.side_effect = queue.Empty()

    with pytest.raises(AudioBlockTimeoutError):
        backend.read_audio_block()
    # The timeout value passed to __init__ must actually be honored by the
    # queue.get() call, not silently ignored.
    backend._audio_queue.get.assert_called_once_with(timeout=0.05)


@patch("backends.vcv_rack.sd.InputStream")
@patch("backends.vcv_rack.mido.open_output")
def test_callback_keeps_only_latest_block_in_bounded_queue(mock_open_output, mock_input_stream, tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text("midi_port_name: 'Fake Port'\nchannels:\n  a:\n    cc: 1\n")
    backend = VCVRackBackend(config_path=str(config_path))

    # Grab the callback VCVRackBackend registered with sd.InputStream(...)
    callback = mock_input_stream.call_args.kwargs["callback"]

    first_block = np.ones((4, 2), dtype=np.float32) * 0.1
    second_block = np.ones((4, 2), dtype=np.float32) * 0.9
    callback(first_block, 4, None, None)
    callback(second_block, 4, None, None)

    # Queue should hold exactly one (the latest) block, not both
    assert backend._audio_queue.qsize() == 1
    result = backend.read_audio_block()
    # read_audio_block mono-mixes stereo -- the *second* (latest) block's
    # values should be what comes back, not the first
    assert np.allclose(result, np.mean(second_block, axis=1))
