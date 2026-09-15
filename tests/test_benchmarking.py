from a_lmi.benchmarking import benchmark_core


def test_benchmark_core_reports_measured_software_results_without_performance_claims():
    report = benchmark_core(iterations=25, records=5, seed=7)

    assert report["classification"] == "local-software-benchmark"
    assert report["errors"] == 0
    assert report["seed"] == 7
    assert report["cst"]["operations"] == 25
    assert report["cst"]["elapsed_seconds"] >= 0.0
    assert report["cst"]["ops_per_second"] >= 0.0
    assert report["continuity"]["records"] == 5
    assert report["continuity"]["verified"] is True
    assert report["continuity"]["restored_memory_records"] == 5
    assert report["continuity"]["bundle_bytes"] > 0
    assert report["continuity"]["elapsed_seconds"] >= 0.0
    assert report["environment"]["python"]
    assert report["environment"]["platform"]
    assert "threshold" not in report
