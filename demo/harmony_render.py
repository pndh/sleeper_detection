"""Offline replica of how vLLM 0.30 renders a chat-completions request for gpt-oss (no server, no GPU).

Mirrors vllm/renderers/online_renderer.py::_make_request_with_harmony and
vllm/entrypoints/openai/parser/harmony_utils.py with default settings
(VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS=0, reasoning_effort unset):
  system  = SystemContent(conversation_start_date)
  developer = instructions from the first system message + function tools
  then user / assistant tool-call / tool messages, rendered for completion as assistant.
Validated against token counts returned by the live server's /tokenize for the 2026-09-29 demo.
Run with the agentdojo env python (it has openai_harmony).
"""
from __future__ import annotations

from openai_harmony import (Author, Conversation, DeveloperContent, HarmonyEncodingName, Message,
                            RenderConversationConfig, Role, SystemContent, TextContent, ToolDescription,
                            load_harmony_encoding)

_ENC = None


def encoding():
    global _ENC
    if _ENC is None:
        _ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _ENC


def to_harmony(request: dict, start_date: str) -> list[Message]:
    msgs = list(request["messages"])
    instructions = None
    if msgs and msgs[0]["role"] in ("system", "developer"):
        instructions = msgs.pop(0)["content"]
    out = [Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_conversation_start_date(start_date))]
    tools = request.get("tools") or []
    if instructions or tools:
        dev = DeveloperContent.new()
        if instructions is not None:
            dev = dev.with_instructions(instructions)
        if tools:
            dev = dev.with_function_tools([ToolDescription.new(name=t["function"]["name"],
                                                               description=t["function"].get("description") or "",
                                                               parameters=t["function"].get("parameters"))
                                           for t in tools])
        out.append(Message.from_role_and_content(Role.DEVELOPER, dev))
    id2name = {tc["id"]: tc["function"]["name"] for m in msgs for tc in (m.get("tool_calls") or [])}
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            if m.get("content"):
                out.append(Message.from_role_and_content(Role.ASSISTANT, m["content"]).with_channel("commentary"))
            for tc in m["tool_calls"]:
                out.append(Message.from_role_and_content(Role.ASSISTANT, tc["function"].get("arguments") or "")
                           .with_channel("commentary").with_recipient(f"functions.{tc['function']['name']}")
                           .with_content_type("json"))
        elif m["role"] == "tool":
            name = id2name.get(m.get("tool_call_id"), "")
            out.append(Message.from_author_and_content(Author.new(Role.TOOL, f"functions.{name}"), m.get("content") or "")
                       .with_channel("commentary").with_recipient("assistant"))
        elif m["role"] == "assistant":
            if m.get("content"):
                out.append(Message.from_role_and_contents(Role.ASSISTANT, [TextContent(text=m["content"])])
                           .with_channel("final"))
        else:
            out.append(Message.from_role_and_contents(Role.USER, [TextContent(text=m.get("content") or "")]))
    return out


def render_tokens(request: dict, start_date: str) -> list[int]:
    conv = Conversation.from_messages(to_harmony(request, start_date))
    return encoding().render_conversation_for_completion(conv, Role.ASSISTANT,
                                                         config=RenderConversationConfig(auto_drop_analysis=False))


def render_text(request: dict, start_date: str) -> str:
    return encoding().decode(render_tokens(request, start_date))


def count_text_tokens(text: str) -> int:
    return len(encoding().encode(text, allowed_special=set()))
