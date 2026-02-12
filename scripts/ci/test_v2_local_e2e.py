from __future__ import annotations

from scripts.ci.run_v2_local_e2e import _wait_ready


def test_wait_ready_times_out_for_closed_port() -> None:
    assert _wait_ready("http://127.0.0.1:9/health", timeout_s=0.3) is False

