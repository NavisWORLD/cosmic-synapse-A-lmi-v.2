from a_lmi.memory.vector_db_client import vector_schema_contract


def test_vector_schema_uses_active_lighttoken_dimensions_and_space_field():
    config = {
        "infrastructure": {"milvus": {"collection_name": "light_tokens_v2"}},
        "a_lmi": {"data_structures": {"embedding_dimension": 1536, "spectral_signature_size": 769}},
    }
    schema = vector_schema_contract(config)
    assert schema["embedding_dimension"] == 1536
    assert schema["spectral_dimension"] == 769
    assert schema["collection_name"] == "light_tokens_v2"
    assert schema["stores_embedding_space"] is True
