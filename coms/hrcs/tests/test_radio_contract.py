from hrcs.physical.radio import RadioModem, stable_data_seed


def test_radio_seed_is_digest_based_and_reproducible():
    assert stable_data_seed(b"COSMOS") == stable_data_seed(b"COSMOS")
    assert stable_data_seed(b"COSMOS") != stable_data_seed(b"cosmos")
    assert 0 <= stable_data_seed(b"COSMOS") < 2**24


def test_frequency_plan_is_deterministic_and_bounded_to_configured_channels():
    modem = RadioModem()
    first = modem.frequency_plan(b"fixture", hop_count=8)
    second = modem.frequency_plan(b"fixture", hop_count=8)
    assert first == second
    assert len(first) == 8
    assert all(freq in modem.channels for freq in first)
    assert modem.frequency_hopping_status == "experimental_tx_only"
