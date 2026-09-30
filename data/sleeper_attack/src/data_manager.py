"""
Data management helpers for loading datasets and writing run results.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InjectionConfig(BaseModel):
    enabled: bool = False
    target_tool: Optional[str] = None
    position: str = "append"
    content: Optional[str] = None


class TargetToolAction(BaseModel):
    tool_name: str
    call_index: int = 1
    position: str = "append"
    content: Optional[str] = None


class SkillInitUpdate(BaseModel):
    name: str
    mode: str
    content: str


class TurnConfig(BaseModel):
    turn_id: int
    user_input: str
    visible_tools: List[str]
    injection_config: InjectionConfig


class MetaInfo(BaseModel):
    domain_src: Optional[str] = None
    domain_tgt: Optional[str] = None
    turn_distance: Optional[int] = None
    semantic_score: Optional[float] = None
    description: Optional[str] = None
    turn_num: Optional[int] = None

    model_config = ConfigDict(extra="ignore")


class TestCase(BaseModel):
    case_id: str
    meta_info: MetaInfo
    visible_tools: List[str]
    memory_init: Optional[Dict[str, Any]] = None
    attacker_info: Optional[Dict[str, Any]] = None
    harmful_goal_categories: List[str] = Field(default_factory=list)
    case_injection: Optional[str] = None
    harmful_goal: Optional[str] = None
    harmful_tools: Optional[List[Dict[str, Any]]] = None
    eval_config: Optional[Dict[str, Any]] = None
    target_tool_actions: List[TargetToolAction] = Field(default_factory=list)
    turns: List[TurnConfig]
    user_prior_knowledge: Optional[str] = None
    ground_truth_params: Optional[List[Dict[str, Any]]] = None
    ask_user_for: Optional[List[str]] = None
    visible_skills: Optional[List[str]] = None
    skill_init_updates: List[SkillInitUpdate] = Field(default_factory=list)


class DataManager:


    def __init__(
        self,
        dataset_path: str,
        result_path: str,
        model_name: Optional[str] = None,
        sample_size: Optional[int] = None,
    ):
        self.dataset_path = dataset_path
        self.result_path = result_path
        self.model_name = model_name
        self.sample_size = sample_size

        result_dir = os.path.dirname(result_path)
        if result_dir and not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)

    def load_test_cases(self, case_ids: Optional[List[str]] = None) -> List[TestCase]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict) and "cases" in raw_data:
            raw_cases = raw_data.get("cases", [])
        elif isinstance(raw_data, list):
            raw_cases = raw_data
        else:
            raise ValueError("Dataset format not supported: expected list or dict with 'cases' key")

        if case_ids:
            wanted = set(case_ids)
            raw_cases = [case for case in raw_cases if case.get("case_id") in wanted]

        if self.sample_size is not None and self.sample_size > 0:
            seed_text = os.getenv("SAMPLE_SEED", "").strip()
            rng = random.Random(int(seed_text)) if seed_text else random
            raw_cases = rng.sample(raw_cases, min(self.sample_size, len(raw_cases)))

        test_cases: List[TestCase] = []
        for item in raw_cases:
            case_visible_tools = item.get("visible_tools", []) or []
            case_injection = item.get("case_injection")
            memory_init = item.get("memory_init")

            turns_raw = item.get("turns", []) or []
            turn_id_list = [
                turn.get("turn_id")
                for turn in turns_raw
                if isinstance(turn, dict) and "turn_id" in turn
            ]
            first_turn_id = turn_id_list[0] if turn_id_list else 1

            if not case_visible_tools:
                aggregated_tools: List[str] = []
                for turn in turns_raw:
                    for tool_name in (turn.get("visible_tools") or []):
                        if tool_name not in aggregated_tools:
                            aggregated_tools.append(tool_name)
                case_visible_tools = aggregated_tools

            target_tool_actions: List[TargetToolAction] = []
            for action in item.get("target_tool_actions", []) or []:
                try:
                    target_tool_actions.append(TargetToolAction(**action))
                except Exception:
                    continue

            injection_plan: Dict[int, InjectionConfig] = {}
            turn_order_map = {idx + 1: turn_id for idx, turn_id in enumerate(turn_id_list)}
            for action in target_tool_actions:
                target_turn_id = turn_order_map.get(action.call_index)
                if target_turn_id is None:
                    target_turn_id = action.call_index if action.call_index in turn_id_list else first_turn_id
                injection_plan[target_turn_id] = InjectionConfig(
                    enabled=True,
                    target_tool=action.tool_name,
                    position=action.position,
                    content=action.content,
                )

            turns: List[TurnConfig] = []
            for turn_data in turns_raw:
                injection_data = turn_data.get("injection_config")
                if injection_data is None:
                    injection_data = injection_plan.get(turn_data.get("turn_id"))

                if isinstance(injection_data, InjectionConfig):
                    injection_config = injection_data
                elif isinstance(injection_data, dict):
                    injection_config = InjectionConfig(**injection_data)
                else:
                    injection_config = InjectionConfig(enabled=False)

                turns.append(
                    TurnConfig(
                        turn_id=turn_data["turn_id"],
                        user_input=turn_data["user_input"],
                        visible_tools=turn_data.get("visible_tools") or case_visible_tools,
                        injection_config=injection_config,
                    )
                )

            meta_info_raw = item.get("meta_info", {}) or {}
            meta_info = MetaInfo(
                domain_src=meta_info_raw.get("domain_src") or meta_info_raw.get("domain_source"),
                domain_tgt=meta_info_raw.get("domain_tgt") or meta_info_raw.get("domain_target"),
                turn_distance=meta_info_raw.get("turn_distance") or meta_info_raw.get("turn_num") or len(turns),
                semantic_score=meta_info_raw.get("semantic_score", 0.0),
                description=meta_info_raw.get("description"),
                turn_num=meta_info_raw.get("turn_num") or len(turns),
            )

            test_cases.append(
                TestCase(
                    case_id=item["case_id"],
                    meta_info=meta_info,
                    visible_tools=case_visible_tools,
                    memory_init=memory_init if isinstance(memory_init, dict) else None,
                    attacker_info=item.get("attacker_info"),
                    harmful_goal_categories=item.get("harmful_goal_categories", []),
                    case_injection=case_injection,
                    harmful_goal=item.get("harmful_goal"),
                    harmful_tools=item.get("harmful_tools", []),
                    eval_config=item.get("eval_config") or item.get("evaluation_config"),
                    target_tool_actions=target_tool_actions,
                    turns=turns,
                    user_prior_knowledge=item.get("user_prior_knowledge"),
                    ground_truth_params=item.get("ground_truth_params"),
                    ask_user_for=item.get("ask_user_for"),
                    visible_skills=item.get("visible_skills") or [],
                    skill_init_updates=item.get("skill_init_updates") or [],
                )
            )

        return test_cases

    def save_result(
        self,
        case_id: str,
        run_logs: List[Dict],
        metrics: Dict,
        test_case_data: Dict = None,
        model_name: Optional[str] = None,
    ):
        resolved_model_name = model_name or self.model_name
        if resolved_model_name is None:
            try:
                from config import AGENT_MODEL as cfg_agent_model

                resolved_model_name = cfg_agent_model
            except Exception:
                resolved_model_name = "unknown-model"

        result_data = {
            "experiment_info": {
                "model_name": resolved_model_name,
            },
            "case_info": {
                "case_id": case_id,
                "test_case_data": test_case_data or {},
            },
            "execution_trajectory": run_logs,
            "evaluation_results": metrics,
            "final_output": {
                "attack_success": metrics.get("attack_success", False),
                "failure_reason": metrics.get("failure_reason"),
                "score": metrics.get("score", 0.0),
            },
        }

        if os.path.exists(self.result_path):
            try:
                with open(self.result_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                    if not isinstance(existing_results, list):
                        existing_results = [existing_results]
            except (json.JSONDecodeError, FileNotFoundError):
                existing_results = []
        else:
            existing_results = []

        existing_results.append(result_data)

        with open(self.result_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)

    def update_result_in_place(
        self,
        case_id: str,
        run_logs: List[Dict],
        metrics: Dict,
        test_case_data: Dict = None,
        model_name: Optional[str] = None,
    ) -> bool:
        resolved_model_name = model_name or self.model_name
        if resolved_model_name is None:
            try:
                from config import AGENT_MODEL as cfg_agent_model

                resolved_model_name = cfg_agent_model
            except Exception:
                resolved_model_name = "unknown-model"

        result_data = {
            "experiment_info": {
                "model_name": resolved_model_name,
            },
            "case_info": {
                "case_id": case_id,
                "test_case_data": test_case_data or {},
            },
            "execution_trajectory": run_logs,
            "evaluation_results": metrics,
            "final_output": {
                "attack_success": metrics.get("attack_success", False),
                "failure_reason": metrics.get("failure_reason"),
                "score": metrics.get("score", 0.0),
            },
        }

        if not os.path.exists(self.result_path):
            return False

        try:
            with open(self.result_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
        except (json.JSONDecodeError, FileNotFoundError):
            return False

        updated = False
        for idx, item in enumerate(existing_results):
            cid = (item.get("case_info") or {}).get("case_id")
            if cid == case_id:
                existing_results[idx] = result_data
                updated = True
                break

        if not updated:
            return False

        with open(self.result_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)
        return True

    def get_result_summary(self) -> Dict[str, Any]:
        if not os.path.exists(self.result_path):
            return {"total_cases": 0, "results": []}

        try:
            with open(self.result_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                if not isinstance(results, list):
                    results = [results]
            return {"total_cases": len(results), "results": results}
        except (json.JSONDecodeError, FileNotFoundError):
            return {"total_cases": 0, "results": []}
