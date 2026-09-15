import pytest

from a_lmi.memory.tkg_client import TKGClient, validate_cypher_identifier


class FakeResult(list):
    def consume(self):
        return None


class FakeSession:
    def __init__(self):
        self.queries = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def run(self, query, **params):
        self.queries.append((query, params))
        if "RETURN elementId(n) AS id" in query:
            return FakeResult([
                {"id": "n1", "labels": ["Person"], "properties": {"name": "Ada"}},
                {"id": "n2", "labels": ["Concept"], "properties": {"name": "COSMOS"}},
            ])
        if "RETURN elementId(a) AS source" in query:
            return FakeResult([
                {"source": "n1", "target": "n2", "type": "STUDIES", "properties": {"timestamp": "2026-09-14"}}
            ])
        return FakeResult()


class FakeDriver:
    def __init__(self):
        self.session_instance = FakeSession()
    def session(self, **kwargs):
        return self.session_instance
    def close(self):
        pass


def config():
    return {"infrastructure": {"neo4j": {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "x", "database": "neo4j"}}}


def test_dynamic_cypher_identifiers_are_strictly_validated():
    assert validate_cypher_identifier("Person_2") == "Person_2"
    with pytest.raises(ValueError):
        validate_cypher_identifier("Person) MATCH (n) DETACH DELETE n //")


def test_fetch_graph_executes_queries_and_returns_real_snapshot_shape():
    driver = FakeDriver()
    client = TKGClient(config(), driver=driver, verify_connection=False)
    snapshot = client.fetch_graph(limit=25)
    assert [n["id"] for n in snapshot["nodes"]] == ["n1", "n2"]
    assert snapshot["nodes"][0]["type"] == "Person"
    assert snapshot["edges"] == [{"source": "n1", "target": "n2", "type": "STUDIES", "properties": {"timestamp": "2026-09-14"}}]
    assert len(driver.session_instance.queries) == 2
