from __future__ import annotations

from anton.governance.policy import PolicyConfig, PolicyEngine


def test_connector_write_requires_approval_by_default():
    engine = PolicyEngine(PolicyConfig())
    decision = engine.evaluate(
        "connector",
        {
            "action": "write",
            "connector_id": "warehouse",
            "query": "DELETE FROM users",
        },
    )
    assert decision.allow is False
    assert decision.requires_approval is True


def test_scratchpad_long_run_requires_approval():
    engine = PolicyEngine(PolicyConfig(max_estimated_seconds_without_approval=30))
    decision = engine.evaluate(
        "scratchpad",
        {
            "action": "exec",
            "estimated_execution_time_seconds": 90,
        },
    )
    assert decision.allow is False
    assert decision.requires_approval is True


def test_blocked_package_install_rejected():
    engine = PolicyEngine(PolicyConfig(blocked_packages={"tensorflow-gpu"}))
    decision = engine.evaluate(
        "scratchpad",
        {
            "action": "install",
            "packages": ["tensorflow-gpu"],
        },
    )
    assert decision.allow is False
    assert decision.requires_approval is False


def test_query_limit_policy_enforced():
    engine = PolicyEngine(PolicyConfig(connector_max_query_limit=100))
    decision = engine.evaluate(
        "connector",
        {
            "action": "query",
            "limit": 200,
        },
    )
    assert decision.allow is False
    assert "exceeds policy" in decision.reason
