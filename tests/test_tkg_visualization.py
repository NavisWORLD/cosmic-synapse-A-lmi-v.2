import pytest

from a_lmi.memory.tkg_client import validate_label
from interfaces.visualization.graph_3d import KnowledgeGraph3D


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        if "elementId(n) AS id" in query:
            return [
                {"id": "n1", "labels": ["Person"], "properties": {"name": "Cory"}},
                {"id": "n2", "labels": ["Concept"], "properties": {"name": "COSMOS"}},
            ]
        if "elementId(a) AS source" in query:
            return [
                {
                    "source": "n1",
                    "target": "n2",
                    "type": "CREATED",
                    "properties": {"timestamp": "2026-09-14T00:00:00Z"},
                }
            ]
        raise AssertionError(f"unexpected query: {query}")


class FakeDriver:
    def session(self):
        return FakeSession()


class FakeNeo4jClient:
    driver = FakeDriver()


def test_dynamic_neo4j_labels_are_validated_before_query_interpolation():
    assert validate_label("LightToken") == "LightToken"
    assert validate_label("PERSON_2") == "PERSON_2"
    with pytest.raises(ValueError):
        validate_label("Person) MATCH (n) DETACH DELETE n //")


def test_visualizer_executes_queries_and_builds_real_graph_data():
    graph = KnowledgeGraph3D()
    graph.load_from_neo4j(FakeNeo4jClient())

    assert [node["id"] for node in graph.nodes] == ["n1", "n2"]
    assert graph.nodes[0]["type"] == "Person"
    assert graph.edges == [
        {
            "source": 0,
            "target": 1,
            "type": "CREATED",
            "properties": {"timestamp": "2026-09-14T00:00:00Z"},
        }
    ]
