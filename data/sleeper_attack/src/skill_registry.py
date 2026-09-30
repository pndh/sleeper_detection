"""Skill registry utilities for file-backed skills."""
import json
import os
import contextvars
from typing import Dict, Any, List, Optional, Callable


def _noop_run(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {}


class Skill:


    def __init__(
        self,
        skill_id: str,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        run: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        body_content: Optional[str] = None,
    ):
        self.skill_id = skill_id
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._run = run
        self.body_content = body_content

    def run(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = context or {}
        return self._run(args, ctx)


class SkillRegistry:


    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._content_overrides: Dict[str, str] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.skill_id] = skill

    def list_skills(self) -> List[Dict[str, Any]]:
        return [
            {
                "skill_id": s.skill_id,
                "name": s.name,
                "description": s.description,
                "input_schema": s.input_schema,
            }
            for s in self._skills.values()
        ]

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def find_skill_id_by_name(self, name: str) -> Optional[str]:

        if not name:
            return None
        name_clean = (name or "").strip()
        if name_clean in self._skills:
            return name_clean
        for sid, s in self._skills.items():
            if (s.name or "").strip() == name_clean:
                return sid
        return None

    def get_display_content(self, skill_id: str) -> Optional[str]:

        if skill_id in self._content_overrides:
            return self._content_overrides[skill_id]
        skill = self.get(skill_id)
        if skill and skill.body_content is not None:
            return skill.body_content
        return None

    def update_skill_content(self, name: str, mode: str, content: str) -> Dict[str, Any]:

        skill_id = self.find_skill_id_by_name(name)
        if not skill_id:
            return {"ok": False, "message": f"Skill not found: {name}", "skill_name": name}
        skill = self.get(skill_id)
        if not skill or skill.body_content is None:
            return {"ok": False, "message": f"Skill {name} has no editable content (code-only skill)", "skill_name": name}
        current = self.get_display_content(skill_id) or ""
        content = (content or "").strip()
        if mode == "cover":
            new_content = content
        elif mode == "append":
            new_content = current + "\n" + content if current else content
        else:
            return {"ok": False, "message": f"Invalid mode: {mode}. Use 'append' or 'cover'.", "skill_name": name}
        self._content_overrides[skill_id] = new_content
        return {"ok": True, "message": f"Updated skill '{name}' ({mode})", "skill_name": name}

    def run(
        self,
        skill_id: str,
        args: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        skill = self.get(skill_id)
        if not skill:
            return {
                "ok": False,
                "error": f"Skill not found: {skill_id}",
                "skill_id": skill_id,
                "result": None,
            }
        try:
            if skill.body_content is not None:
                display = self.get_display_content(skill_id)
                result = {"content": display if display is not None else ""}
            else:
                result = skill.run(args or {}, context or {})
            return {
                "ok": True,
                "skill_id": skill_id,
                "result": result,
                "error": None,
            }
        except Exception as e:
            return {
                "ok": False,
                "skill_id": skill_id,
                "result": None,
                "error": str(e),
            }


def _skill_todo_from_request(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:

    task = (args.get("task_description") or args.get("task") or "").strip()
    if not task:
        return {"todos": [], "message": "No task_description provided."}

    parts = [p.strip() for p in task.replace(";", "\n").replace("，", ",").split("\n") if p.strip()]
    if not parts:
        parts = [task]
    todos = [{"id": i + 1, "text": p} for i, p in enumerate(parts)]
    return {"todos": todos, "count": len(todos)}


def _skill_eval_config_draft(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:

    scenario = (args.get("scenario") or args.get("description") or "").strip()
    if not scenario:
        return {"eval_type": "keyword_match", "criteria": {"keywords": []}, "message": "No scenario given."}
    return {
        "eval_type": "keyword_match",
        "criteria": {
            "keywords": [w.strip() for w in scenario.split() if len(w.strip()) > 1][:5],
        },
        "scenario_summary": scenario[:200],
    }


def _skill_summarize_memory_usage(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:

    category = (args.get("category") or "").strip()
    return {
        "message": "Use MemoryRetrieve(query='ALL') to review saved notes, or pass a short text query to search relevant notes.",
        "categories_hint": "memory is note-based in the current setting; save new notes with MemoryUpdate(mode, content).",
        "requested_category": category or "(any)",
    }


def get_default_registry() -> SkillRegistry:

    r = SkillRegistry()
    r.register(Skill(
        skill_id="todo_from_request",
        name="Todo from request",
        description="Turn a user task description into a structured TODO list (one item per line or comma).",
        input_schema={
            "type": "object",
            "properties": {
                "task_description": {"type": "string", "description": "The user's task or request text"},
            },
            "optional": ["task_description"],
        },
        run=_skill_todo_from_request,
    ))
    r.register(Skill(
        skill_id="eval_config_draft",
        name="Eval config draft",
        description="Generate a minimal eval_config draft (eval_type and criteria) for a given scenario.",
        input_schema={
            "type": "object",
            "properties": {
                "scenario": {"type": "string", "description": "Short scenario or description"},
            },
            "optional": ["scenario"],
        },
        run=_skill_eval_config_draft,
    ))
    r.register(Skill(
        skill_id="summarize_memory_usage",
        name="Summarize memory usage",
        description="Return brief guidance on how to use MemoryRetrieve/MemoryUpdate for a category.",
        input_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Memory category (e.g. contacts, notes)"},
            },
            "optional": ["category"],
        },
        run=_skill_summarize_memory_usage,
    ))
    return r


def load_skills_from_dir(skill_data_root: str, skill_ids: List[str]) -> SkillRegistry:


    registry = SkillRegistry()
    for sid in skill_ids or []:
        sid = (sid or "").strip()
        if not sid:
            continue
        dir_path = os.path.join(skill_data_root, sid)
        meta_path = os.path.join(dir_path, "metadata.json")
        body_path = os.path.join(dir_path, "skill.md")
        if not os.path.isdir(dir_path):
            print(f"[skill_registry] Skip {sid}: not a directory: {dir_path}")
            continue
        name = sid
        description = ""
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                name = meta.get("name") or name
                description = meta.get("description") or ""
            except Exception as e:
                print(f"[skill_registry] Skip {sid}: failed to read metadata.json: {e}")
        body_content = ""
        if os.path.isfile(body_path):
            try:
                with open(body_path, "r", encoding="utf-8") as f:
                    body_content = f.read()
            except Exception as e:
                print(f"[skill_registry] Skip {sid}: failed to read skill.md: {e}")
        skill = Skill(
            skill_id=sid,
            name=name,
            description=description,
            input_schema={"type": "object", "properties": {}, "optional": []},
            run=_noop_run,
            body_content=body_content,
        )
        registry.register(skill)
    return registry


_default_registry: Optional[SkillRegistry] = None


_registry_var: contextvars.ContextVar[Optional[SkillRegistry]] = contextvars.ContextVar(
    "skill_registry", default=None
)


def get_registry() -> SkillRegistry:

    ctx_reg = _registry_var.get()
    if ctx_reg is not None:
        return ctx_reg
    global _default_registry
    if _default_registry is None:
        _default_registry = get_default_registry()
    return _default_registry


def set_registry(registry: SkillRegistry) -> None:

    _registry_var.set(registry)


def reset_registry() -> None:

    _registry_var.set(None)
    global _default_registry
    _default_registry = None
