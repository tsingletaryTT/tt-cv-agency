# tests/test_vcv_rack_backend.py
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from backends.vcv_rack import VCVRackBackend, load_channel_config


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
