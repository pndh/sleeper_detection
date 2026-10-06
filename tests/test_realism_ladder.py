"""Tests for realism ladder dataset (docs/pivot_plan.md W3).

Run: .venv-tools/bin/python -m pytest tests/test_realism_ladder.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "interp" / "data" / "realism_ladder"

pytestmark = pytest.mark.skipif(not (DS / "labels.jsonl").exists(), reason="realism_ladder not built")

LABEL_KEYS = {"realism_level", "realism_name", "is_harmful", "twin_id", "domain_name", "case_id", "payload_text"}


def rows(name):
    return [json.loads(line) for line in open(DS / f"{name}.jsonl")]


def test_defender_view_has_no_label_fields():
    for r in rows("requests"):
        assert not (LABEL_KEYS & set(r)), set(r) & LABEL_KEYS
        assert re.fullmatch(r"[0-9a-f]{16}", r["id"])
    assert {r["id"] for r in rows("requests")} == {r["id"] for r in rows("labels")}


def test_realism_levels_and_twins():
    labs = rows("labels")
    assert len(labs) == 100
    by_case = defaultdict(lambda: defaultdict(list))
    for lab in labs:
        by_case[lab["case_id"]][lab["realism_level"]].append(lab)

    for case_id, lvls in by_case.items():
        assert set(lvls.keys()) == {0, 1, 2, 3, 4}
        for lvl, items in lvls.items():
            assert len(items) == 2
            harmful = [it for it in items if it["is_harmful"]]
            benign = [it for it in items if not it["is_harmful"]]
            assert len(harmful) == 1 and len(benign) == 1
            assert harmful[0]["twin_id"] == benign[0]["id"]
            assert benign[0]["twin_id"] == harmful[0]["id"]


def test_variants_differ_only_in_memory_note():
    reqs = {r["id"]: r["request"] for r in rows("requests")}
    labs = rows("labels")
    by_case = defaultdict(list)
    for lab in labs:
        by_case[lab["case_id"]].append(lab)

    for case_id, case_labs in by_case.items():
        ref_req = reqs[case_labs[0]["id"]]
        for lab in case_labs:
            curr = reqs[lab["id"]]
            # System prompt, user query, tool call name must match
            assert curr["messages"][0] == ref_req["messages"][0]
            assert curr["messages"][1] == ref_req["messages"][1]
            assert curr["messages"][2]["tool_calls"] == ref_req["messages"][2]["tool_calls"]
            assert curr["tools"] == ref_req["tools"]
