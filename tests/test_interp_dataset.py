"""Checks on interp/data/<dataset>. Run: cd ~/sleeper_detection && .venv-tools/bin/python -m pytest -q tests
PLAN(2) §3/§12: detectors get only the request; labels live in a separate evaluation-side file."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "interp" / "data" / "demo_v1"
LABEL_KEYS = {"variant", "poisoned", "hard_negative", "turn_type", "case_id", "strategy", "payload_entry_index",
              "extra_note_entry_index", "behavior", "query_name", "source", "label", "labels"}


def rows(name):
    return [json.loads(l) for l in open(DS / f"{name}.jsonl")]


def test_defender_view_has_no_label_fields():
    for r in rows("requests"):
        assert not (LABEL_KEYS & set(r)), set(r) & LABEL_KEYS
        assert re.fullmatch(r"[0-9a-f]{16}", r["id"]), "ids must be opaque"
        text = json.dumps(r["request"])
        assert "poisoned" not in text and "benign_equivalent" not in text


def test_every_request_has_a_label():
    assert {r["id"] for r in rows("requests")} == {r["id"] for r in rows("labels")}


def test_type_b_twins_same_length_and_differ_only_in_extra_note():
    req = {r["id"]: r for r in rows("requests")}
    lab = rows("labels")
    by_q = {}
    for l in lab:
        if l["source"] == "type_b" and l["variant"] in ("poisoned", "benign_equivalent"):
            by_q.setdefault(l["query_name"], {})[l["variant"]] = l
    assert by_q
    for q, pair in by_q.items():
        p, b = req[pair["poisoned"]["id"]], req[pair["benign_equivalent"]["id"]]
        assert p["n_tokens"] == b["n_tokens"], q
        k = pair["poisoned"]["extra_note_entry_index"]
        inside = {t for a, z in p["entry_spans"][k]["occurrences"] for t in range(a, z)}
        diff = {i for i, (x, y) in enumerate(zip(p["input_ids"], b["input_ids"])) if x != y}
        assert diff, q
        assert diff <= inside, f"{q}: tokens differ outside the note spans: {sorted(diff - inside)[:10]}"


def test_entry_spans_decode_to_entry_text():
    sys.path.insert(0, str(ROOT / "demo"))
    import harmony_render as hr
    enc = hr.encoding()
    for r in rows("requests"):
        for e in r["entry_spans"]:
            assert e["occurrences"], e["text"]
            for a, z in e["occurrences"]:
                assert e["text"][:40] in enc.decode(r["input_ids"][a:z]).replace("\\'", "'")
