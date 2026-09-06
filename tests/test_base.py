import numpy as np
import pytest
from backends.base import CVBackend, FakeCVBackend


def test_fake_backend_records_set_cv_and_tracks_last_known():
    backend = FakeCVBackend(channel_names=["a", "b"])
    assert backend.channels() == ["a", "b"]
    assert backend.last_known_cv("a") == 0.5  # documented default before any set_cv
    backend.set_cv("a", 0.9)
    assert backend.last_known_cv("a") == 0.9
    assert backend.last_known_cv("b") == 0.5


def test_fake_backend_rejects_unknown_channel():
    backend = FakeCVBackend(channel_names=["a"])
    with pytest.raises(KeyError):
        backend.set_cv("nonexistent", 0.5)


def test_fake_backend_read_audio_block_returns_configured_block():
    block = np.ones(512, dtype=np.float32) * 0.25
    backend = FakeCVBackend(channel_names=["a"], audio_block=block)
    result = backend.read_audio_block()
    assert np.array_equal(result, block)


def test_cv_backend_is_abstract():
    with pytest.raises(TypeError):
        CVBackend()


def test_fake_backend_reports_default_sample_rate_and_block_size():
    backend = FakeCVBackend(channel_names=["a"])
    # Defaults match VCVRackBackend's real production values, so a test
    # written against FakeCVBackend without overriding these exercises the
    # same numbers the real backend would report.
    assert backend.sample_rate() == 48000
    assert backend.block_size() == 1024


def test_fake_backend_reports_configured_sample_rate_and_block_size():
    backend = FakeCVBackend(channel_names=["a"], sample_rate=44100, block_size=512)
    assert backend.sample_rate() == 44100
    assert backend.block_size() == 512
