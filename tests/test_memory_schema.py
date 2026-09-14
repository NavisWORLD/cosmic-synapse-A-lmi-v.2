from a_lmi.memory.schema import vector_field_dimensions


def test_vector_store_dimensions_match_lighttoken_contract():
    assert vector_field_dimensions() == {
        "joint_embedding": 1536,
        "spectral_power": 769,
    }
