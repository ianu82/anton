from __future__ import annotations

from anton.minion.registry import MinionInfo, MinionRegistry, MinionStatus


class TestMinionStatus:
    def test_status_values(self):
        assert MinionStatus.PENDING.value == "pending"
        assert MinionStatus.RUNNING.value == "running"
        assert MinionStatus.COMPLETED.value == "completed"
        assert MinionStatus.FAILED.value == "failed"
        assert MinionStatus.KILLED.value == "killed"


class TestMinionInfo:
    def test_make_id(self):
        id1 = MinionInfo.make_id()
        id2 = MinionInfo.make_id()
        assert len(id1) == 12
        assert id1 != id2

    def test_default_status(self):
        m = MinionInfo(id="test", task="do stuff", folder="/tmp")
        assert m.status == MinionStatus.PENDING
        assert m.pid is None
        assert m.error is None
        assert m.completed_at is None
        assert m.cron_expr is None

    def test_with_cron(self):
        m = MinionInfo(
            id="test", task="periodic check", folder="/tmp",
            cron_expr="*/5 * * * *"
        )
        assert m.cron_expr == "*/5 * * * *"


class TestMinionRegistry:
    def test_register_and_get(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="task1", folder="/tmp")
        registry.register(m)

        assert registry.get("m1") is m
        assert registry.get("nonexistent") is None

    def test_list_all(self):
        registry = MinionRegistry()
        m1 = MinionInfo(id="m1", task="task1", folder="/tmp")
        m2 = MinionInfo(id="m2", task="task2", folder="/tmp")
        registry.register(m1)
        registry.register(m2)

        assert len(registry.list_all()) == 2

    def test_list_running(self):
        registry = MinionRegistry()
        m1 = MinionInfo(id="m1", task="t1", folder="/tmp", status=MinionStatus.RUNNING)
        m2 = MinionInfo(id="m2", task="t2", folder="/tmp", status=MinionStatus.PENDING)
        registry.register(m1)
        registry.register(m2)

        running = registry.list_running()
        assert len(running) == 1
        assert running[0].id == "m1"

    def test_list_scheduled(self):
        registry = MinionRegistry()
        m1 = MinionInfo(id="m1", task="t1", folder="/tmp", cron_expr="* * * * *")
        m2 = MinionInfo(id="m2", task="t2", folder="/tmp")
        registry.register(m1)
        registry.register(m2)

        scheduled = registry.list_scheduled()
        assert len(scheduled) == 1
        assert scheduled[0].id == "m1"

    def test_update_status(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="task1", folder="/tmp")
        registry.register(m)

        result = registry.update_status("m1", MinionStatus.RUNNING)
        assert result is True
        assert registry.get("m1").status == MinionStatus.RUNNING

    def test_update_status_nonexistent(self):
        registry = MinionRegistry()
        assert registry.update_status("nope", MinionStatus.RUNNING) is False

    def test_update_status_completed_sets_timestamp(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="t1", folder="/tmp")
        registry.register(m)
        assert m.completed_at is None

        registry.update_status("m1", MinionStatus.COMPLETED)
        assert m.completed_at is not None

    def test_update_status_failed_with_error(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="t1", folder="/tmp")
        registry.register(m)

        registry.update_status("m1", MinionStatus.FAILED, error="boom")
        assert m.status == MinionStatus.FAILED
        assert m.error == "boom"
        assert m.completed_at is not None

    def test_killed_clears_cron(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="t1", folder="/tmp", cron_expr="*/5 * * * *")
        registry.register(m)

        registry.update_status("m1", MinionStatus.KILLED)
        assert m.status == MinionStatus.KILLED
        assert m.cron_expr is None  # Cron cleared when killed
        assert m.completed_at is not None

    def test_remove(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="t1", folder="/tmp")
        registry.register(m)

        assert registry.remove("m1") is True
        assert registry.get("m1") is None
        assert registry.remove("m1") is False

    def test_remove_nonexistent(self):
        registry = MinionRegistry()
        assert registry.remove("nope") is False
