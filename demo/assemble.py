"""Assemble paired API requests (poisoned twin / clean twin) from a Sleeper Attack case.

Defender = API provider: the agent's memory exists only as text inside the request.
Every piece of the request is produced by the Sleeper Attack paper's own released code
(data/sleeper_attack/src), so the request matches what the paper's agent runtime would send:

  system   : AgentRunner._get_agent_instructions(case)          (paper's system prompt)
  tools    : tool_schemas.json, exported from the paper's @function_tool definitions via the Agents SDK
  user     : one benign query
  assistant: MemoryRetrieve(query="ALL")                          (paper instructs the agent to do this first)
  tool     : str(_memory_retrieve_impl("ALL")) over the memory store (the SDK serializes tool dicts with str())

Clean twin memory    = case.memory_init
Poisoned twin memory = case.memory_init + the exact note the attack plants (MemoryUpdate content in
                       target_tool_actions), i.e. the store after a successful plant turn.

Deviation from the paper's main runs (documented): the paper keeps one session across turns, so its
trigger-turn request still contains the turn-1 web page with the injection. Here the session is fresh
and only memory carries the payload, matching the threat model and the paper's `fresh_session_replay`
supplement.
"""
from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "data" / "sleeper_attack"
sys.path.insert(0, str(PAPER))

from src import virtual_tools as vt  # noqa: E402
from src.agent_runner import AgentRunner  # noqa: E402
from src.data_manager import DataManager  # noqa: E402

TOOL_SCHEMAS = json.loads((PAPER / "tool_schemas.json").read_text())
MEMORY_CALL_ID = "call_memory_retrieve_0"


def load_case(strategy: str, state: str, case_id: str | None = None, index: int = 0):
    dm = DataManager(str(PAPER / "datasets" / strategy / f"{state}.json"), tempfile.mktemp(suffix=".json"),
                     model_name="demo")
    cases = dm.load_test_cases()
    if case_id is None:
        return cases[index]
    return next(c for c in cases if c.case_id == case_id)


def system_prompt(case) -> str:
    class _Self:  # the method only needs _has_memory_tools from self
        pass
    s = _Self()
    s._has_memory_tools = lambda vis: AgentRunner._has_memory_tools(s, vis)
    return AgentRunner._get_agent_instructions(s, case, case.visible_tools)


def planted_note(case) -> str:
    """The exact memory note a successful plant turn writes (MemoryUpdate content in the injection)."""
    for action in case.target_tool_actions or []:
        m = re.search(r'MemoryUpdate with mode="append" and content="(.*?)"\.\s', action.content or "", re.S)
        if m:
            return m.group(1)
    raise ValueError(f"{case.case_id}: no MemoryUpdate(append) payload found in target_tool_actions")


def memory_tool_output(entries: list[str]) -> str:
    vt.set_memory_store({"entries": list(entries)})
    return str(vt._memory_retrieve_impl(query="ALL"))


def build_request(case, benign_query: str, poisoned: bool) -> dict:
    entries = list((case.memory_init or {}).get("entries", []))
    if poisoned:
        entries.append(planted_note(case))  # appended last, so the paper's store shows it as most recent
    return {
        "messages": [
            {"role": "system", "content": system_prompt(case)},
            {"role": "user", "content": benign_query},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": MEMORY_CALL_ID, "type": "function",
                "function": {"name": "MemoryRetrieve", "arguments": json.dumps({"query": "ALL"})}}]},
            {"role": "tool", "tool_call_id": MEMORY_CALL_ID, "content": memory_tool_output(entries)},
        ],
        "tools": [{"type": "function", "function": TOOL_SCHEMAS[n]["function"]}
                  for n in case.visible_tools if n in TOOL_SCHEMAS],
    }


def build_twins(case, benign_query: str) -> dict:
    twins = {"clean": build_request(case, benign_query, poisoned=False),
             "poisoned": build_request(case, benign_query, poisoned=True)}
    return copy.deepcopy(twins)


def twin_diff(twins: dict) -> dict:
    """Which messages differ between twins, and is the payload the only thing added."""
    c, p = twins["clean"], twins["poisoned"]
    differing = [i for i, (a, b) in enumerate(zip(c["messages"], p["messages"])) if a != b]
    return {
        "same_message_count": len(c["messages"]) == len(p["messages"]),
        "same_tools": c["tools"] == p["tools"],
        "differing_message_indices": differing,
        "differing_roles": [c["messages"][i]["role"] for i in differing],
        "only_memory_tool_result_differs": differing == [3],
    }
