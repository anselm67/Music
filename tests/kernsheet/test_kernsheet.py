"""Unit tests for KernSheet catalog management — score/entry deletion."""

import json
from pathlib import Path

import pytest

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


def test_training_sample_counts_tolerates_unloadable_layout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A score whose layout JSON won't parse must be skipped, not abort the whole
    # report. _make writes a non-JSON stub ("x") for the layout, so load_score
    # raises for this score; the helper should log-and-continue and still print.
    from cli.kernsheet import _training_sample_counts

    ks = _make(tmp_path, {"work/p01": [{"json_path": "work/p01.json"}]})
    _training_sample_counts(ks)  # must not raise

    assert "Training samples" in capsys.readouterr().out


def test_load_score_stamps_canonical_id_over_embedded(tmp_path: Path) -> None:
    # The on-disk `id` is non-authoritative; load_score must return the catalog
    # id, not whatever (possibly stale/wrong) id the file embeds.
    ks = _make(tmp_path, {"work/p08": [{"json_path": "work/p08-catelin.json"}]})
    (tmp_path / "layout/work/p08-catelin.json").write_text(
        json.dumps({"id": "work/p09-catelin", "pages": []})  # WRONG embedded id
    )

    assert ks.load_score("work/p08-catelin").id == "work/p08-catelin"


def test_delete_score_with_mismatched_key_is_a_noop(tmp_path: Path) -> None:
    # Passing a score that belongs to a DIFFERENT entry (e.g. from a worklist
    # poisoned by a bad embedded id) must not unlink its layout or mutate either
    # entry — otherwise the file goes but the catalog keeps a dangling reference.
    ks = _make(
        tmp_path,
        {
            "work/p08": [{"json_path": "work/p08-catelin.json"}],
            "work/p09": [{"json_path": "work/p09-catelin.json"}],
        },
    )

    ks.delete_score("work/p08", ks.id2score["work/p09-catelin"])

    assert (tmp_path / "layout/work/p09-catelin.json").exists()  # NOT unlinked
    assert ks.has_score("work/p09-catelin")  # still a live score
    assert [s.json_path for s in ks.catalog.entries["work/p09"].scores] == [
        "work/p09-catelin.json"
    ]
    assert [s.json_path for s in ks.catalog.entries["work/p08"].scores] == [
        "work/p08-catelin.json"
    ]


def test_delete_persists_to_catalog_on_disk(tmp_path: Path) -> None:
    ks = _make(tmp_path, {"work": [{"json_path": "work/ed0.json"}]})

    ks.delete_entry("work")

    # A freshly loaded catalog no longer has the entry.
    assert "work" not in KernSheet(tmp_path).catalog.entries
