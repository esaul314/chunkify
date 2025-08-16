from pathlib import Path

import pytest


@pytest.mark.xfail(reason="run_report emission is implemented in later stories", strict=True)
def test_run_report_placeholder():
    # Placeholder: will assert existence when Story E2 adds run_report.json
    assert Path("run_report.json").exists()
