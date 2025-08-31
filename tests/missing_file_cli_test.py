from __future__ import annotations

from tests.scripts_cli_test import _run_cli


def test_convert_missing_file_exits_nonzero() -> None:
    result = _run_cli("convert", "missing.pdf")
    assert result.returncode != 0
    err = result.stderr.lower()
    assert "does not exist" in err or "no such file" in err
