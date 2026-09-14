import numpy as np

from a_lmi.security.federated import FederatedLearning


def test_weighted_average_is_named_for_what_it_does_without_torch_dependency():
    updates = [
        {"w": np.array([1.0, 3.0], dtype=np.float32)},
        {"w": np.array([5.0, 7.0], dtype=np.float32)},
    ]
    result = FederatedLearning.weighted_average(updates, weights=[0.25, 0.75])
    np.testing.assert_allclose(result["w"], np.array([4.0, 6.0], dtype=np.float32))


def test_privacy_properties_do_not_claim_formal_dp_or_secure_aggregation():
    properties = FederatedLearning.privacy_properties()
    assert properties["formal_differential_privacy"] is False
    assert properties["cryptographic_secure_aggregation"] is False
    assert properties["experimental_noise_injection"] is True


def test_experimental_noise_is_reproducible_with_seed_and_not_labeled_dp():
    update = {"w": np.ones(4, dtype=np.float32)}
    first = FederatedLearning.add_experimental_gaussian_noise(update, stddev=0.1, seed=9)
    second = FederatedLearning.add_experimental_gaussian_noise(update, stddev=0.1, seed=9)
    np.testing.assert_array_equal(first["w"], second["w"])
