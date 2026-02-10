from src.monitoring.dashboard import RiskMonitor, build_dashboard_payload


def test_risk_monitor_update_and_status():
    monitor = RiskMonitor(window=5)
    monitor.update(0.2)
    monitor.update(0.8)
    assert monitor.status(threshold=0.7) == "alert"


def test_build_dashboard_payload():
    payload = build_dashboard_payload(
        timestamps=["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"],
        risk_scores=[0.3, 0.9],
        threshold=0.7,
    )
    assert payload["n_alerts"] == 1
    assert payload["series"][1]["alert"]
