"""Checks on interp/data/behavior_v1 (demo/build_behavior_set.py). Run: .venv-tools/bin/python -m pytest -q tests"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "interp" / "data" / "behavior_v1"
sys.path.insert(0, str(ROOT / "demo"))
pytestmark = pytest.mark.skipif(not (DS / "labels.jsonl").exists(), reason="behavior_v1 not built")
LABEL_KEYS = {"framing", "template_id", "split", "judge", "domain", "action", "note_text", "case_id", "behavior"}


def rows(name):
    return [json.loads(line) for line in open(DS / f"{name}.jsonl")]


def test_defender_view_has_no_label_fields():
    for r in rows("requests"):
        assert not (LABEL_KEYS & set(r)), set(r) & LABEL_KEYS
        assert re.fullmatch(r"[0-9a-f]{16}", r["id"])
    assert {r["id"] for r in rows("requests")} == {r["id"] for r in rows("labels")}


def test_templates_per_framing_and_no_cue_words():
    import build_behavior_set as B
    assert len(B.LIVE_TEMPLATES) >= 5 and len(B.INERT_TEMPLATES) >= 5
    by_case = defaultdict(lambda: defaultdict(set))
    for lab in rows("labels"):
        by_case[lab["case_id"]][lab["framing"]].add(lab["template_id"])
        if lab["note_text"]:
            assert not B.CUE_WORDS.search(lab["note_text"]), lab["note_text"]
    for case, f in by_case.items():
        assert f["live"] == set(range(len(B.LIVE_TEMPLATES))) and f["inert"] == set(range(len(B.INERT_TEMPLATES))), case
        assert f["none"] == {None}, case


def test_payload_tool_is_callable_and_query_is_the_trigger_turn():
    req = {r["id"]: r for r in rows("requests")}
    for lab in rows("labels"):
        r = req[lab["id"]]["request"]
        assert lab["judge"]["target_tool"] in {t["function"]["name"] for t in r["tools"]}, lab["case_id"]
        assert r["messages"][1]["role"] == "user"


def test_variants_of_a_case_differ_only_inside_the_note():
    req = {r["id"]: r for r in rows("requests")}
    by_case = defaultdict(list)
    for lab in rows("labels"):
        by_case[lab["case_id"]].append(lab)
    for case, labs in by_case.items():
        ref = req[next(l["id"] for l in labs if l["framing"] == "none")]["request"]
        for lab in labs:
            r = req[lab["id"]]
            assert r["request"]["messages"][:3] == ref["messages"][:3] and r["request"]["tools"] == ref["tools"], case
            if lab["note_text"]:
                assert lab["extra_note_entry_index"] == 0, case  # newest entry
                assert r["entry_spans"][0]["text"] == lab["note_text"]


def test_split_holds_out_domains_and_templates_together():
    import build_behavior_set as B
    m = json.loads((DS / "manifest.json").read_text())["preregistered"]
    test_doms = set(m["test_domains"])
    assert test_doms and not (test_doms & set(m["never_test_domains"]))
    seen = defaultdict(set)
    for lab in rows("labels"):
        d_test, t_test = lab["domain"] in test_doms, lab["template_id"] in B.TEST_TEMPLATE_IDS
        if lab["split"] == "train":
            assert not d_test and not t_test
        elif lab["split"] == "test":
            assert d_test and (t_test or lab["template_id"] is None)
        seen[lab["split"]].add(lab["domain"])
    assert not (seen["train"] & seen["test"])
