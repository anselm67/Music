"""Unit tests for KernSheet catalog management — score/entry deletion."""

import json
from pathlib import Path

from kernsheet import KernSheet


def _catalog(tmp_path: Path, entries: dict[str, list[dict[str, str]]]) -> None:
    cat: dict[str, object] = {"entries": {}}
    entries_obj = cat["entries"]
    assert isinstance(entries_obj, dict)
    for key, scores in entries.items():
        entries_obj[key] = {
            "source": "",
            "imslp_query": "",
            "imslp_url": "",
            "scores": [
                {
                    "pdf_path": s.get("pdf_path", ""),
                    "pdf_url": "",
                    "json_path": s["json_path"],
                }
                for s in scores
            ],
        }
    (tmp_path / "catalog.json").write_text(json.dumps(cat))


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def _make(
    tmp_path: Path,
    entries: dict[str, list[dict[str, str]]],
    *,
    pdf: str | None = None,
) -> KernSheet:
    """Write the catalog plus the kern/tokens/layout (and optional pdf) files for
    every entry, and return a KernSheet over it."""
    _catalog(tmp_path, entries)
    for key, scores in entries.items():
        for s in scores:
            _touch(tmp_path / "layout" / s["json_path"])
        _touch(tmp_path / f"{key}.krn")
        _touch(tmp_path / "build" / "tokens" / f"{key}.tokens")
    if pdf:
        _touch(tmp_path / pdf)
    return KernSheet(tmp_path)


def test_delete_score_keeps_entry_and_shared_files_when_not_last(
    tmp_path: Path,
) -> None:
    ks = _make(
        tmp_path,
        {"work": [{"json_path": "work/ed0.json"}, {"json_path": "work/ed1.json"}]},
    )

    ks.delete_score("work", ks.id2score["work/ed0"])

    assert not (tmp_path / "layout/work/ed0.json").exists()  # own layout gone
    assert (tmp_path / "layout/work/ed1.json").exists()  # sibling kept
    assert (tmp_path / "work.krn").exists()  # shared kern kept
    assert (tmp_path / "build/tokens/work.tokens").exists()
    assert [s.json_path for s in ks.catalog.entries["work"].scores] == ["work/ed1.json"]
    assert not ks.has_score("work/ed0") and ks.has_score("work/ed1")


def test_delete_last_score_cascades_to_delete_entry(tmp_path: Path) -> None:
    ks = _make(tmp_path, {"work": [{"json_path": "work/ed0.json"}]})

    ks.delete_score("work", ks.id2score["work/ed0"])

    assert not (tmp_path / "layout/work/ed0.json").exists()
    assert not (tmp_path / "work.krn").exists()
    assert not (tmp_path / "build/tokens/work.tokens").exists()
    assert "work" not in ks.catalog.entries


def test_delete_entry_removes_owned_files_but_keeps_shared_pdf(tmp_path: Path) -> None:
    ks = _make(
        tmp_path,
        {
            "work": [
                {"json_path": "work/ed0.json", "pdf_path": "shared.pdf"},
                {"json_path": "work/ed1.json", "pdf_path": "shared.pdf"},
            ]
        },
        pdf="shared.pdf",
    )

    ks.delete_entry("work")

    assert not (tmp_path / "layout/work/ed0.json").exists()
    assert not (tmp_path / "layout/work/ed1.json").exists()
    assert not (tmp_path / "work.krn").exists()
    assert not (tmp_path / "build/tokens/work.tokens").exists()
    assert (tmp_path / "shared.pdf").exists()  # shared pdf left untouched
    assert "work" not in ks.catalog.entries
    assert not ks.has_score("work/ed0") and not ks.has_score("work/ed1")
    assert "work/ed1" not in ks.id2key


def test_delete_persists_to_catalog_on_disk(tmp_path: Path) -> None:
    ks = _make(tmp_path, {"work": [{"json_path": "work/ed0.json"}]})

    ks.delete_entry("work")

    # A freshly loaded catalog no longer has the entry.
    assert "work" not in KernSheet(tmp_path).catalog.entries
