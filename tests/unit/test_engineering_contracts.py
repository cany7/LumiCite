from __future__ import annotations

from pathlib import Path


def _src_files() -> list[Path]:
    return sorted(Path("src").rglob("*.py"))


def test_no_raw_print_calls_in_src():
    offenders: list[str] = []

    for path in _src_files():
        text = path.read_text(encoding="utf-8")
        if "print(" in text:
            offenders.append(str(path))

    assert offenders == []


def test_main_has_no_import_star():
    main_path = Path("src/main.py")
    text = main_path.read_text(encoding="utf-8")

    assert " import *" not in text
