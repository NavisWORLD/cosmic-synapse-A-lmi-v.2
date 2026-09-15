import numpy as np

from a_lmi.memory.vector_db_client import VectorDBClient, vector_schema_contract


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


class _LoadRequiredCollection:
    def __init__(self):
        self.loaded = False
        self.load_calls = 0
        self.search_calls = 0

    def load(self):
        self.loaded = True
        self.load_calls += 1

    def search(self, **kwargs):
        assert self.loaded, "collection must be loaded before search"
        self.search_calls += 1
        return []


def _client_with_load_required_collection():
    client = VectorDBClient.__new__(VectorDBClient)
    client.collection = _LoadRequiredCollection()
    client.config = {"metric_type": "L2"}
    client.contract = {
        "embedding_dimension": 1536,
        "spectral_dimension": 769,
    }
    return client


def test_semantic_search_loads_collection_before_query():
    client = _client_with_load_required_collection()

    assert client.search_semantic(np.zeros(1536, dtype=np.float32)) == []
    assert client.collection.load_calls == 1
    assert client.collection.search_calls == 1


def test_spectral_search_loads_collection_before_query():
    client = _client_with_load_required_collection()

    assert client.search_spectral(np.zeros(769, dtype=np.float32)) == []
    assert client.collection.load_calls == 1
    assert client.collection.search_calls == 1
