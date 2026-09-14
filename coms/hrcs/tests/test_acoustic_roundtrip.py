import numpy as np

from hrcs.physical.acoustic import AcousticModem


def test_synthetic_acoustic_round_trip_without_hardware():
    modem = AcousticModem(stochastic_noise_level=0.0)
    payload = b"COSMOS"
    signal = modem.modulate(payload)
    assert signal.ndim == 1
    assert modem.demodulate(signal) == payload


def test_synthetic_acoustic_round_trip_tolerates_small_channel_noise():
    modem = AcousticModem(stochastic_noise_level=0.0)
    payload = b"synapse"
    signal = modem.modulate(payload)
    noisy = signal + np.random.default_rng(42).normal(0.0, 0.002, signal.shape)
    assert modem.demodulate(noisy) == payload
