from a_lmi.reasoning.hypothesis_generator import HypothesisGenerator


class FakeSearchProvider:
    name = "fixture-search"

    def search(self, query):
        return [
            {
                "title": "Observed source",
                "uri": "https://example.test/evidence/1",
                "snippet": "fixture evidence",
            }
        ]


def test_hypothesis_separates_observation_from_speculation_and_records_provenance():
    generator = HypothesisGenerator(
        config_source={"infrastructure": {}},
        driver=None,
        search_provider=FakeSearchProvider(),
    )
    pattern = {
        "type": "correlation",
        "entity_a": "audio energy",
        "entity_b": "x12 response",
        "cooccurrence": 4,
    }

    hypothesis = generator.generate_hypothesis(pattern)

    assert hypothesis["classification"] == "hypothesis"
    assert hypothesis["observation"]["type"] == "correlation"
    assert hypothesis["speculation"] is True
    assert 0.0 <= hypothesis["uncertainty"] <= 1.0
    assert hypothesis["evidence_inputs"] == [pattern]
    assert hypothesis["provenance"]["search_provider"] == "fixture-search"
    assert hypothesis["investigation_results"][0]["uri"].endswith("/1")
    assert "example.com/search" not in str(hypothesis)


def test_no_search_provider_returns_queries_not_fake_urls():
    generator = HypothesisGenerator(config_source={"infrastructure": {}}, driver=None)
    hypothesis = generator.generate_hypothesis(
        {"type": "knowledge_gap", "entity_a": {"name": "A"}, "entity_b": {"name": "B"}}
    )
    assert hypothesis["investigation_results"] == []
    assert hypothesis["investigation_queries"]
    assert all(not query.startswith("http") for query in hypothesis["investigation_queries"])
