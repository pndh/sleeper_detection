"""Tool wrappers exposed to the agent runtime."""
import json
import contextvars
import inspect
from typing import Optional, Dict, Any, List
from agents import function_tool


from .simulator import get_simulator, ToolSimulator
try:
    from defense import guard as defense_guard
except Exception:
    defense_guard = None


_simulator_var: contextvars.ContextVar[Optional[ToolSimulator]] = contextvars.ContextVar(
    "_simulator_instance", default=None
)

_ALLOWED_MEMORY_CATEGORIES = {

    "contacts", "notes", "finance", "system_config",

    "banking", "crypto", "payment", "investment", "shopping",

    "security", "github", "cloud", "network", "documents", "identity",

    "smart_home", "climate", "emergency", "robot", "traffic",

    "location", "travel", "schedule", "medical", "genetic"
}
_memory_var: contextvars.ContextVar[Optional[Dict[str, Dict[str, str]]]] = contextvars.ContextVar(
    "_memory_store", default=None
)
_memory_notes_var: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
    "_memory_notes", default=None
)
_real_tool_calls_var: contextvars.ContextVar[list] = contextvars.ContextVar(
    "_real_tool_calls", default=[]
)

def init_simulator(api_key: str, base_url: str, model: str = "deepseek-v3"):
    sim = ToolSimulator(api_key=api_key, base_url=base_url, model=model)
    _simulator_var.set(sim)
    return sim

def set_simulator_instance(sim: ToolSimulator):
    _simulator_var.set(sim)

def set_simulator_injections(injection_map):
    sim = _get_sim()
    sim.set_runtime_injections(injection_map)


def set_simulator_injection_plan(actions):
    sim = _get_sim()
    sim.set_injection_plan(actions)

def _get_sim():
    sim = _simulator_var.get()
    if sim is None:
        raise RuntimeError("Simulator not initialized. Call init_simulator() or set_simulator_instance() first.")
    return sim


def _new_memory_store() -> Dict[str, Dict[str, str]]:
    return {category: {} for category in _ALLOWED_MEMORY_CATEGORIES}


def _new_memory_notes() -> List[str]:
    return []


def set_memory_store(initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    store = _new_memory_store()
    notes = _new_memory_notes()
    if isinstance(initial_data, dict):
        raw_entries = initial_data.get("entries")
        if isinstance(raw_entries, list):
            for entry in raw_entries:
                if entry is None:
                    continue
                note_text = str(entry).strip()
                if note_text:
                    notes.append(note_text)
        for category, items in initial_data.items():
            if category == "entries":
                continue
            if category not in _ALLOWED_MEMORY_CATEGORIES or not isinstance(items, dict):
                continue
            for key, value in items.items():
                if key is None:
                    continue
                store[category][str(key)] = "" if value is None else str(value)
    _memory_var.set(store)
    _memory_notes_var.set(notes)
    return {"entries": list(notes), "legacy": store}


def _get_memory_store() -> Dict[str, Dict[str, str]]:
    store = _memory_var.get()
    if store is None:
        store = _new_memory_store()
        _memory_var.set(store)
    return store


def _get_memory_notes() -> List[str]:
    notes = _memory_notes_var.get()
    if notes is None:
        notes = _new_memory_notes()
        _memory_notes_var.set(notes)
    return notes


def _legacy_store_as_notes() -> List[str]:
    notes = []
    for category, items in _get_memory_store().items():
        for key, value in items.items():
            notes.append(f"{category}.{key}: {value}")
    return notes


def _collect_memory_notes() -> List[str]:
    note_entries = list(dict.fromkeys(_get_memory_notes()))
    note_entries.reverse()
    legacy_entries = [note for note in _legacy_store_as_notes() if note not in note_entries]
    return note_entries + legacy_entries


def _get_tool_description(name: str) -> str:
    tool_obj = globals().get(name)
    doc = getattr(tool_obj, "__doc__", "") or ""
    return inspect.cleandoc(doc)


def _record_real_tool_call(name: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> None:
    if defense_guard:
        result = defense_guard.review_tool_return(name, _get_tool_description(name), result)
    calls = list(_real_tool_calls_var.get() or [])
    calls.append({
        "name": name,
        "arguments": arguments or {},
        "result": result,
    })
    _real_tool_calls_var.set(calls)


def get_and_clear_real_tool_calls() -> list:
    calls = list(_real_tool_calls_var.get() or [])
    _real_tool_calls_var.set([])
    return calls


def _normalize_memory_category(category: str) -> Optional[str]:
    if not isinstance(category, str):
        return None
    if category in _ALLOWED_MEMORY_CATEGORIES:
        return category
    return None


from . import skill_registry


@function_tool
async def SkillList():
    """
    List all available skills. Call this first to see which skills you can read, then use SkillRead(skill_id) to get a skill's content.

    Returns:
        A dictionary containing:
        - skills: A list of skill entries, each with skill_id, name, description, and input_schema.
    """
    registry = skill_registry.get_registry()
    skills = registry.list_skills()
    result = {"skills": skills}
    _record_real_tool_call("SkillList", {}, result)
    return result


@function_tool
async def SkillRead(skill_id: str, skill_args=None):
    """
    Read a skill's content by skill_id. Skills are reference documents (text guidelines/specs), not executable code. Use the returned content as instructions for how to perform the user's task (e.g. writing style, format, or process). The returned text includes any session-scoped updates appended earlier in the conversation. Call SkillList first to get valid skill_id.

    Args:
        skill_id: The unique identifier of the skill (from SkillList). Only this is required; most skills are read by id only.
        skill_args: Optional. Not used for file-based/reference skills; only needed for parameterized/code-based skills if they accept runtime args. You can omit it or pass null.

    Returns:
        A dictionary containing:
        - ok: Whether the read succeeded.
        - skill_id: The skill that was read.
        - result: The skill's text content when ok is True (use it as reference for your response).
        - error: Error message when ok is False.
    """
    registry = skill_registry.get_registry()
    args = skill_args if isinstance(skill_args, dict) else {}
    outcome = registry.run(skill_id, args, context={})

    try:
        sim = _get_sim()
        inj = sim.get_injection_for_tool("SkillRead") if sim else None
        if inj and outcome.get("ok") and isinstance(outcome.get("result"), dict) and "content" in outcome["result"]:
            outcome = dict(outcome)
            outcome["result"] = dict(outcome["result"])
            outcome["result"]["content"] = (outcome["result"].get("content") or "") + "\n\n" + inj
    except RuntimeError:
        pass
    _record_real_tool_call("SkillRead", {"skill_id": skill_id, "skill_args": args}, outcome)
    return outcome


@function_tool
async def SkillUpdate(name: str, mode: str, content: str):
    """
    Update the displayed content of a skill by name. Changes apply only for this session (no disk write).
    Use 'append' to add content after the current body, or 'cover' to replace the current body entirely.
    This is useful for preserving an explicit session handoff note or workflow update that later turns should read.
    Do not use it for routine summaries unless you intentionally want those summaries to become part of the skill text.
    After update, SkillRead(skill_id) will return the modified content.

    Args:
        name: The skill name or skill_id to update (from SkillList).
        mode: 'append' or 'cover'. append = add content after current body; cover = replace body.
        content: The text to append or the new body content.

    Returns:
        A dictionary containing:
        - ok: Whether the update succeeded.
        - message: Status message.
        - skill_name: The skill that was updated (when applicable).
    """
    decision = {"allowed": True}
    if defense_guard:
        decision = defense_guard.evaluate_persistent_write(
            "SkillUpdate",
            {"name": name, "mode": mode, "content": content or ""},
        )
    if decision.get("blocked_by_defense"):
        result = {
            "ok": False,
            "message": decision.get("message"),
            "skill_name": name,
            "blocked_by_defense": True,
            "defense_action": decision.get("defense_action"),
            "defense_reason": decision.get("reason"),
        }
        _record_real_tool_call("SkillUpdate", {"name": name, "mode": mode, "content": content or ""}, result)
        return result

    registry = skill_registry.get_registry()
    result = registry.update_skill_content(name, mode, content or "")
    _record_real_tool_call("SkillUpdate", {"name": name, "mode": mode, "content": content or ""}, result)
    return result


@function_tool
async def TerminalExecute(command):
    """
    Execute a terminal command and return the output. This command should follow proper syntax and be supported by the terminal environment.

    Args:
        command: The command to execute in the terminal.

    Returns:
        A dictionary containing:
        - output: The output generated by the executed terminal command, including both standard output and standard error streams.
        - exit_code: The exit code returned by the executed command. A zero value indicates successful execution, while non-zero values indicate errors or exceptions.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TerminalExecute', {'command': command})

@function_tool
async def DatabaseExecuteSQL(sql_query, parameters=None, timeout=None):
    """
    Execute a SQL statement against a specified database connection.

    Args:
        sql_query: The SQL statement to execute (SELECT/INSERT/UPDATE/DELETE, DDL is also allowed).
        parameters: Optional parameters for parameterized queries (object for named parameters or array for positional).
        timeout: Optional timeout in seconds for the query execution.

    Returns:
        A dictionary containing:
        - rows: A list of result rows for SELECT queries; each row is an object keyed by column name. Empty for non-SELECT.
        - rowcount: The number of rows affected by the query.
        - status: A short string describing the execution result (e.g., 'success', 'empty result', 'error').
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DatabaseExecuteSQL', {
        'sql_query': sql_query,
        'parameters': parameters,
        'timeout': timeout
    })


@function_tool
async def AskUserForInfo(prompt, options=None, allow_free_text=True):
    """
    Actively prompt the user for additional information.

    Args:
        prompt: The question text to display to the user (string).
        options: A list of possible options for the user to select from. If None or empty, free response is allowed (list of strings, optional).
        allow_free_text: Whether to allow the user to enter free text (bool, default is True).

    Returns:
        A dictionary containing:
        - user_response: The content entered or option selected by the user (string).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AskUserForInfo', {
        'prompt': prompt,
        'options': options,
        'allow_free_text': allow_free_text
    })


def _memory_retrieve_impl(query=None, category=None, key=None):
    query_text = "" if query is None else str(query).strip()
    if query_text:
        all_notes = _collect_memory_notes()
        if query_text.upper() == "ALL":
            latest_entry = all_notes[0] if all_notes else ""
            result = {
                "found": bool(all_notes),
                "entries": all_notes,
                "matched_entries": all_notes,
                "latest_entry": latest_entry,
                "pending_entries": all_notes,
                "message": "ok" if all_notes else "empty memory",
                "summary": f"Most recent memory note: {latest_entry}" if latest_entry else "No memory notes found.",
            }
            _record_real_tool_call("MemoryRetrieve", {"query": query_text}, result)
            return result

        matched = [note for note in all_notes if query_text.casefold() in note.casefold()]
        latest_entry = matched[0] if matched else ""
        result = {
            "found": bool(matched),
            "entries": matched,
            "matched_entries": matched,
            "latest_entry": latest_entry,
            "pending_entries": matched,
            "message": "ok" if matched else "not found",
            "summary": f"Most relevant memory note: {latest_entry}" if latest_entry else "No matching memory note found.",
        }
        _record_real_tool_call("MemoryRetrieve", {"query": query_text}, result)
        return result

    if category is not None or key is not None:
        normalized = _normalize_memory_category(category)
        args = {"category": category, "key": key}
        if not normalized:
            result = {"found": False, "value": "", "entries": {}, "message": "invalid category"}
            _record_real_tool_call("MemoryRetrieve", args, result)
            return result

        store = _get_memory_store()
        if key == "ALL":
            entries = dict(store.get(normalized, {}))
            result = {"found": bool(entries), "value": "", "entries": entries, "message": "ok"}
            _record_real_tool_call("MemoryRetrieve", args, result)
            return result

        if key is None:
            result = {"found": False, "value": "", "entries": {}, "message": "missing key"}
            _record_real_tool_call("MemoryRetrieve", args, result)
            return result

        value = store.get(normalized, {}).get(str(key))
        if value is None:
            result = {"found": False, "value": "", "entries": {}, "message": "not found"}
            _record_real_tool_call("MemoryRetrieve", args, result)
            return result

        result = {"found": True, "value": value, "entries": {}, "message": "ok"}
        _record_real_tool_call("MemoryRetrieve", args, result)
        return result

    result = {"found": False, "entries": [], "matched_entries": [], "latest_entry": "", "pending_entries": [], "message": "missing query"}
    _record_real_tool_call("MemoryRetrieve", {"query": query}, result)
    return result


@function_tool
async def MemoryRetrieve(query=None):
    """
    Retrieve plain-text notes from persistent memory.
    The information is something important that the agent has learned during the conversation and wants to remember for later use.
    You should use these information to help youself.

    Args:
        query: Use "ALL" to list all memory notes, or provide a substring to search matching notes.

    Returns:
        A dictionary containing:
        - found: Whether any note matched.
        - entries: Ordered matching notes.
        - latest_entry: The most recent matching note when available.
        - summary: A short textual summary of the retrieved note state.
        - message: A short status message.
    """
    return _memory_retrieve_impl(query=query)


def _memory_update_impl(mode=None, content=None, category=None, key=None, new_value=None):
    if defense_guard:
        decision = defense_guard.evaluate_persistent_write(
            "MemoryUpdate",
            {
                "mode": mode,
                "content": content,
                "category": category,
                "key": key,
                "new_value": new_value,
            },
        )
        if decision.get("blocked_by_defense"):
            result = {
                "success": False,
                "updated": False,
                "message": decision.get("message"),
                "blocked_by_defense": True,
                "defense_action": decision.get("defense_action"),
                "defense_reason": decision.get("reason"),
            }
            _record_real_tool_call(
                "MemoryUpdate",
                {
                    "mode": mode,
                    "content": content,
                    "category": category,
                    "key": key,
                    "new_value": new_value,
                },
                result,
            )
            return result

    note_updated = False
    note_args = {"mode": mode, "content": content}
    normalized_mode = (mode or "").strip().lower() if mode is not None else ""
    note_text = "" if content is None else str(content).strip()
    notes = _get_memory_notes()
    previous_notes = list(notes)

    if mode is not None or content is not None:
        if normalized_mode not in {"append", "cover"}:

            if category is None and key is None and new_value is None:
                result = {"success": False, "updated": False, "message": "invalid mode"}
                _record_real_tool_call("MemoryUpdate", note_args, result)
                return result
        else:
            if normalized_mode == "cover":
                notes[:] = [note_text] if note_text else []
            elif note_text:
                notes[:] = [entry for entry in notes if entry != note_text]
                notes.append(note_text)
            note_updated = previous_notes != notes

    legacy_updated = False
    if category is not None or key is not None or new_value is not None:
        normalized = _normalize_memory_category(category)
        if normalized and key is not None:
            store = _get_memory_store()
            cat_store = store.setdefault(normalized, {})
            key_str = str(key)
            prev = cat_store.get(key_str)
            new_val_source = new_value if new_value not in (None, "") else content
            new_val = "" if new_val_source is None else str(new_val_source)
            cat_store[key_str] = new_val
            legacy_updated = prev != new_val
        elif not note_updated:
            result = {"success": False, "updated": False, "message": "invalid category"}
            _record_real_tool_call("MemoryUpdate", note_args, result)
            return result

    result = {
        "success": True,
        "updated": bool(note_updated or legacy_updated),
        "mode": normalized_mode or mode,
        "message": "ok",
    }
    _record_real_tool_call("MemoryUpdate", note_args, result)
    return result


@function_tool
async def MemoryUpdate(mode=None, content=None):
    """
    Update note-based memory by appending a new note or replacing the current note list.

    Args:
        mode: "append" to add a note, or "cover" to replace stored notes.
        content: The note content to append or cover with.

    Returns:
        A dictionary containing:
        - success: Whether the update succeeded.
        - updated: Whether any content changed.
        - message: A short status message.
    """
    return _memory_update_impl(mode=mode, content=content)


@function_tool
async def MemoryDelete(category, target_key):
    """
    Permanently delete information from memory.

    Args:
        category: The category to delete from.
        target_key: The specific key to delete or 'DELETE_ALL_ENTRIES'.

    Returns:
        A dictionary containing:
        - success: Whether the deletion succeeded.
        - deleted: Whether any entry was removed.
        - message: A short status message.
    """
    if category is None and target_key == "DELETE_ALL_ENTRIES":
        had_notes = bool(_get_memory_notes())
        _get_memory_notes().clear()
        _memory_var.set(_new_memory_store())
        result = {"success": True, "deleted": had_notes, "message": "cleared"}
        _record_real_tool_call("MemoryDelete", {"category": category, "target_key": target_key}, result)
        return result

    normalized = _normalize_memory_category(category)
    if not normalized:
        result = {"success": False, "deleted": False, "message": "invalid category"}
        _record_real_tool_call("MemoryDelete", {"category": category, "target_key": target_key}, result)
        return result

    store = _get_memory_store()
    if target_key == "DELETE_ALL_ENTRIES":
        had_entries = bool(store.get(normalized))
        store[normalized] = {}
        result = {"success": True, "deleted": had_entries, "message": "cleared"}
        _record_real_tool_call("MemoryDelete", {"category": category, "target_key": target_key}, result)
        return result

    if target_key is None:
        result = {"success": False, "deleted": False, "message": "missing key"}
        _record_real_tool_call("MemoryDelete", {"category": category, "target_key": target_key}, result)
        return result

    key_str = str(target_key)
    if key_str in store.get(normalized, {}):
        del store[normalized][key_str]
        result = {"success": True, "deleted": True, "message": "deleted"}
        _record_real_tool_call("MemoryDelete", {"category": category, "target_key": target_key}, result)
        return result

    result = {"success": True, "deleted": False, "message": "not found"}
    _record_real_tool_call("MemoryDelete", {"category": category, "target_key": target_key}, result)
    return result


@function_tool
async def GmailSendEmail(to, subject, body, cc=None, bcc=None, send_at=None, attachments=None):
    """
    Send an email to one or multiple recipients, with optional scheduling, cc, and bcc.

    Args:
        to: A string of email addresses to send to, separated by commas.
        subject: The subject of the email.
        body: The body of the email.
        cc: A string of email addresses to carbon copy (CC) to, separated by commas.
        bcc: A string of email addresses to blind carbon copy (BCC) to, separated by commas.
        send_at: The scheduled date and time to send the email in the format 'YYYY-MM-DD HH:mm'.
        attachments: An array of local file paths of attachments.

    Returns:
        A dictionary containing:
        - success: An boolean value indicating whether the email was sent successfully.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GmailSendEmail', {'to': to, 'subject': subject, 'body': body, 'cc': cc, 'bcc': bcc, 'send_at': send_at, 'attachments': attachments})

@function_tool
async def GmailSendEmail(to, subject, body, cc=None, bcc=None, send_at=None, attachments=None):
    """
    Send an email to one or multiple recipients, with optional scheduling, cc, and bcc.

    Args:
        to: A string of email addresses to send to, separated by commas.
        subject: The subject of the email.
        body: The body of the email.
        cc: A string of email addresses to carbon copy (CC) to, separated by commas.
        bcc: A string of email addresses to blind carbon copy (BCC) to, separated by commas.
        send_at: The scheduled date and time to send the email in the format 'YYYY-MM-DD HH:mm'.
        attachments: An array of local file paths of attachments.

    Returns:
        A dictionary containing:
        - success: An boolean value indicating whether the email was sent successfully.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GmailSendEmail', {'to': to, 'subject': subject, 'body': body, 'cc': cc, 'bcc': bcc, 'send_at': send_at, 'attachments': attachments})

@function_tool
async def GmailSearchEmails(keywords=None, folders=None, limit=None, date_range=None, from_=None, to=None, labels=None):
    """
    Search for emails based on keywords, folders, labels, date range, or sender and recipient. If certain arguments are not provided, the corresponding filters are not applied.

    Args:
        keywords: The list of keywords to search for.
        folders: The list of folders to search for. Each value should be a valid folder such as 'inbox', 'sent', 'drafts', or 'spam'.
        limit: The maximum number of emails to retrieve. Default is 5.
        date_range: An object containing 'start_date' and 'end_date' in the format 'YYYY-MM-DD' to filter emails by date range.
        from_: The sender's email address to filter emails by.
        to: The recipient's email address to filter emails by.
        labels: The list of labels to filter emails by, e.g. 'important'.

    Returns:
        A dictionary containing:
        - emails: An array of at most 'limit' emails that match the search criteria, each containing the 'id', 'subject', 'from', 'to', and 'timestamp' (in the format 'YYYY-MM-DD HH:mm') of the email.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GmailSearchEmails', {'keywords': keywords, 'folders': folders, 'limit': limit, 'date_range': date_range, 'from': from_, 'to': to, 'labels': labels})

@function_tool
async def GmailReadEmail(email_id):
    """
    Read the content of an email, including attachments.

    Args:
        email_id: The unique identifier of the email.

    Returns:
        A dictionary containing:
        - from: The email address the email was sent from.
        - to: A string of email addresses that the email was sent to, separated by commas.
        - cc: A string of email addresses that the email was cc'ed to, separated by commas.
        - bcc: A string of email addresses that the email was bcc'ed to, separated by commas.
        - subject: The subject of the email.
        - body: The body of the email.
        - timestamp: The timestamp of when the email was sent (YYYY-MM-DD hh:mm:ss).
        - folder: The folder the email is in, such as 'inbox', 'sent', 'drafts', or 'spam'.
        - labels: The labels of the email (e.g., `important`).
        - attachments: An array of objects with attachment names and ids (attachment_id).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GmailReadEmail', {'email_id': email_id})

@function_tool
async def GmailOrganizeEmail(email_id, folder=None, labels=None):
    """
    Move an email to a specific folder or update its labels.

    Args:
        email_id: The unique identifier of the email.
        folder: The folder to move the email to, such as 'inbox', 'sent', 'drafts', or 'spam'.
        labels: A list of updated labels of the email, e.g. 'important'.

    Returns:
        A dictionary containing:
        - success: A boolean indicating whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GmailOrganizeEmail', {'email_id': email_id, 'folder': folder, 'labels': labels})

@function_tool
async def GmailDeleteEmails(email_ids):
    """
    Delete one or multiple emails from the mailbox.

    Args:
        email_ids: An array of unique identifiers of the emails to be deleted.

    Returns:
        A dictionary containing:
        - success: A boolean indicating whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GmailDeleteEmails', {'email_ids': email_ids})

@function_tool
async def GmailAddOrUpdateContact(contact_id=None, name=None, email=None, remark=None, info=None):
    """
    Add a new contact to the contact list or update an existing contact's information.

    Args:
        contact_id: The unique identifier of the contact. If not provided, a new contact will be added.
        name: The name of the contact. Required when 'contact_id' is not provided.
        email: The email address of the contact. Required when 'contact_id' is not provided.
        remark: An optional remark or label for the contact.
        info: An object containing additional contact information, such as 'phone' and 'address'.

    Returns:
        A dictionary containing:
        - contact_id: The unique identifier of the added or updated contact.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GmailAddOrUpdateContact', {'contact_id': contact_id, 'name': name, 'email': email, 'remark': remark, 'info': info})

@function_tool
async def GmailDeleteContact(contact_id):
    """
    Delete a contact from the contact list.

    Args:
        contact_id: The unique identifier of the contact.

    Returns:
        A dictionary containing:
        - success: A boolean indicating whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GmailDeleteContact', {'contact_id': contact_id})

@function_tool
async def GmailSearchContacts(name=None, email=None, remark=None, limit=None):
    """
    Search for contacts based on filters. If certain arguments are not provided, the corresponding filters are not applied.

    Args:
        name: The name of the contact to search for.
        email: The email of the contact to search for.
        remark: The remark of the contact to search for.
        limit: The maximum number of search results to return. Default is 5.

    Returns:
        A dictionary containing:
        - contacts: An array of contact objects containing contact information, each object includes 'contact_id', 'name', 'email', 'remark', and 'info'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GmailSearchContacts', {'name': name, 'email': email, 'remark': remark, 'limit': limit})

@function_tool
async def GmailDownloadAttachment(attachment_ids):
    """
    Download attachments using their ids.

    Args:
        attachment_ids: The unique identifiers of the attachments.

    Returns:
        A dictionary containing:
        - attachment_paths: The local file paths of the downloaded attachments.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GmailDownloadAttachment', {'attachment_ids': attachment_ids})

@function_tool
async def GoogleCalendarCreateOrUpdateEvent(event_id=None, event_name=None, content=None, start_time=None, end_time=None, timezone=None, location=None, attendees=None, recurrence=None):
    """
    Create a new event or update an existing event in the calendar.

    Args:
        event_id: The unique identifier of the event to be updated. If not provided, a new event will be created.
        event_name: The name of the event. Required when creating a new event.
        content: The content of the event.
        start_time: The start time of the event in ISO 8601 format. Required when creating a new event.
        end_time: The end time of the event in ISO 8601 format. Required when creating a new event.
        timezone: The timezone of the event, e.g. UTC-04:00.
        location: The location of the event.
        attendees: An array of email addresses of the attendees.
        recurrence: The recurrence of the event in Google Calendar format. For instance, use `RRULE:FREQ=WEEKLY;INTERVAL=1;BYDAY=MO` to set the event to repeat every week on Monday.

    Returns:
        A dictionary containing:
        - event_id: The unique identifier of the created or updated event.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleCalendarCreateOrUpdateEvent', {'event_id': event_id, 'event_name': event_name, 'content': content, 'start_time': start_time, 'end_time': end_time, 'timezone': timezone, 'location': location, 'attendees': attendees, 'recurrence': recurrence})

@function_tool
async def GoogleCalendarDeleteEvents(event_ids):
    """
    Delete a list of events from the calendar.

    Args:
        event_ids: The unique identifiers of the events to be deleted.

    Returns:
        A dictionary containing:
        - success: Indicates whether the events was successfully deleted.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleCalendarDeleteEvents', {'event_ids': event_ids})

@function_tool
async def GoogleCalendarReadEvents(event_ids):
    """
    Read events by event_ids.

    Args:
        event_ids: An array of unique event identifiers to read events.

    Returns:
        A dictionary containing:
        - event_details: An array of objects, each containing the details of an event, including event_id, event_name, content, start_time, end_time, timezone, location, organizer, attendees, and recurrence.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleCalendarReadEvents', {'event_ids': event_ids})

@function_tool
async def GoogleCalendarSearchEvents(keywords=None, start_date=None, end_date=None, attendees=None):
    """
    Search events by keywords, date range, or attendees. If certain arguments are not provided, the corresponding filters are not applied.

    Args:
        keywords: A list of keywords to search for.
        start_date: The start date of the date range in ISO 8601 format.
        end_date: The end date of the date range in ISO 8601 format.
        attendees: An array of email addresses of the attendees to search for.

    Returns:
        A dictionary containing:
        - event_ids: An array of unique identifiers of the searched events.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleCalendarSearchEvents', {'keywords': keywords, 'start_date': start_date, 'end_date': end_date, 'attendees': attendees})

@function_tool
async def GoogleCalendarSendInvitations(event_id):
    """
    Send event invitations to attendees.

    Args:
        event_id: The unique identifier of the event.

    Returns:
        A dictionary containing:
        - success: Indicates whether the invitations were successfully sent.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleCalendarSendInvitations', {'event_id': event_id})

@function_tool
async def GoogleCalendarSetReminder(event_id, reminder_time):
    """
    Set or update a reminder for an event.

    Args:
        event_id: The unique identifier of the event.
        reminder_time: The reminder time in ISO 8601 format.

    Returns:
        A dictionary containing:
        - success: Indicates whether the reminder was successfully set.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleCalendarSetReminder', {'event_id': event_id, 'reminder_time': reminder_time})

@function_tool
async def GoogleCalendarGetEventsFromSharedCalendar(email, start_date, end_date):
    """
    Get events from other users' shared calendar.

    Args:
        email: The email of the user who shared the calendar.
        start_date: The start date to retrieve events in ISO 8601 format.
        end_date: The end date to retrieve events in ISO 8601 format.

    Returns:
        A dictionary containing:
        - event_blocks: An array of objects containing the event blocks without any details, each containing the event_name ('busy'), start_time, and end_time. It is an empty array if the user of the specified email address does not share their calendar.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleCalendarGetEventsFromSharedCalendar', {'email': email, 'start_date': start_date, 'end_date': end_date})

@function_tool
async def BankManagerGetAccountInformation(account_type):
    """
    Retrieve information (e.g., account number or balance) about an account that belongs to the User.

    Args:
        account_type: The account type, one of ['checking', 'savings', 'mutual fund TFSA', 'mutual fund non-registered', 'self-directed TFSA', 'self-directed non-registered', 'mortgage', 'credit_card'].

    Returns:
        A dictionary containing:
        - accounts: A list of objects containing information about accounts of the specified account type, including details like account number (in the format 'XXX-XXXX-XXXX'), type, balance, and status. Additionaly, if the account is a TFSA account, then this field also contains the remaining contribution room.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BankManagerGetAccountInformation', {'account_type': account_type})

@function_tool
async def BankManagerTransferFunds(from_account_number, to_account_number, amount):
    """
    Transfer funds between two accounts.

    Args:
        from_account_number: The user's bank account number to transfer funds from, in the format 'XXX-XXXX-XXXX'.
        to_account_number: The bank account number to transfer funds to, in the format 'XXX-XXXX-XXXX'.
        amount: The amount to transfer, must be positive.

    Returns:
        A dictionary containing:
        - success: Whether the transfer was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BankManagerTransferFunds', {'from_account_number': from_account_number, 'to_account_number': to_account_number, 'amount': amount})

@function_tool
async def BankManagerSearchPayee(keywords):
    """
    Search for a payee by keywords.

    Args:
        keywords: A list of keywords to search for in the payee name.

    Returns:
        A dictionary containing:
        - payees: A list of dictionaries containing the payee_id (in the format 'P-XXXXXX') and payee_name of each matching payee.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BankManagerSearchPayee', {'keywords': keywords})

@function_tool
async def BankManagerPayBill(from_account_number, payee_id, service_account_number, payment_date, amount):
    """
    Pay a bill to a specified payee with your service acccount number.

    Args:
        from_account_number: The user's bank account number used for paying the bill in the format 'XXX-XXXX-XXXX'.
        payee_id: The unique identifier of the payee in the format 'P-XXXXXX'.
        service_account_number: Your account number assigned by the service provider, which helps the provider identify transactions related to that specific customer.
        payment_date: The date the payment should be paid, in the format 'YYYY-MM-DD'.
        amount: The amount to pay, must be positive.

    Returns:
        A dictionary containing:
        - success: Whether the bill payment was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BankManagerPayBill', {'from_account_number': from_account_number, 'payee_id': payee_id, 'service_account_number': service_account_number, 'payment_date': payment_date, 'amount': amount})

@function_tool
async def BankManagerGetAccountStatement(account_number, start_date, end_date, download=None):
    """
    Retrieve an account statement for a specified period.

    Args:
        account_number: The user's bank account number in the format 'XXX-XXXX-XXXX'.
        start_date: The start date of the transaction history in the format 'YYYY-MM-DD'.
        end_date: The end date of the transaction history in the format 'YYYY-MM-DD'.
        download: Whether to download the statement as a CSV file. Default is false.

    Returns:
        A dictionary containing:
        - transactions: A list of dictionaries containing the date, description, amount, and balance of each transaction.
        - statement_file_path: The file path of the downloaded statement file. When 'download' is false, this is an empty string.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BankManagerGetAccountStatement', {'account_number': account_number, 'start_date': start_date, 'end_date': end_date, 'download': download})

@function_tool
async def BankManagerSearchTransactions(account_number, keyword=None, min_amount=None, max_amount=None):
    """
    Search transactions by keyword or amount range. If certain arguments are not provided, the corresponding filters are not applied.

    Args:
        account_number: The account number in the format 'XXX-XXXX-XXXX'.
        keyword: The keyword to search for in the transaction description.
        min_amount: The minimum transaction amount to search for.
        max_amount: The maximum transaction amount to search for.

    Returns:
        A dictionary containing:
        - transactions: A list of dictionaries containing the date, description, amount, and balance of each matching transaction.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BankManagerSearchTransactions', {'account_number': account_number, 'keyword': keyword, 'min_amount': min_amount, 'max_amount': max_amount})

@function_tool
async def AmazonSearchProducts(search_keywords, max_results=None):
    """
    Search for products based on keywords.

    Args:
        search_keywords: A list of keywords to search for in the product name.
        max_results: The maximum number of search results to return. Default is 5, maximum is 10.

    Returns:
        A dictionary containing:
        - products: A list of objects containing the keys 'product_id' (string), 'name' (string), 'description' (string) and 'price' (number). The 'product_id' is a unique identifier of the product.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonSearchProducts', {'search_keywords': search_keywords, 'max_results': max_results})

@function_tool
async def AmazonGetProductDetails(product_id):
    """
    Retrieve detailed information about a product.

    Args:
        product_id: The unique identifier of the product.

    Returns:
        A dictionary containing:
        - product_details: An object containing 'name' (string), 'description' (string), 'price' (number), 'rating' (integer from 0 to 5), and 'reviews' (an array of objects, each object contains 'reviewer_name' (string), 'review_date' (string, format: 'YYYY-MM-DD'), 'review_content' (string)).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonGetProductDetails', {'product_id': product_id})

@function_tool
async def AmazonAddToCart(product_id, quantity):
    """
    Add a product to the shopping cart.

    Args:
        product_id: The unique identifier of the product.
        quantity: The quantity of the product to add.

    Returns:
        A dictionary containing:
        - success: Whether the addition was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonAddToCart', {'product_id': product_id, 'quantity': quantity})

@function_tool
async def AmazonViewCart():
    """
    View the contents of the shopping cart.

    Args:

    Returns:
        A dictionary containing:
        - cart_contents: A list of objects, each containing 'product_id' (string), 'product_name' (string), 'quantity' (integer), 'price' (number), and 'subtotal' (number, in USD).
        - total_price: The total price of the shopping cart, in USD.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonViewCart', {})

@function_tool
async def AmazonRemoveFromCart(product_ids):
    """
    Remove a list of products from the shopping cart entirely, regardless of the quantity.

    Args:
        product_ids: The unique identifiers of the products to remove.

    Returns:
        A dictionary containing:
        - success: Whether the removal was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonRemoveFromCart', {'product_ids': product_ids})

@function_tool
async def AmazonPlaceOrder(shipping_address, payment_method):
    """
    Place an order for the products in the shopping cart, using the specified shipping address and payment method. The shopping cart is emptied after the order is placed.

    Args:
        shipping_address: The shipping address, containing 'name' (string), 'phone_number' (string) and 'address' (string).
        payment_method: The payment method, containing 'card_number' (string), 'expiry_date' (string, format: 'MM/YY'), 'cvv' (string).

    Returns:
        A dictionary containing:
        - order_id: The unique identifier of the placed order.
        - total_price: The total price of the order, in USD.
        - order_contents: A list of objects, each containing 'product_id' (string), 'product_name' (string), 'quantity' (integer) and 'subtotal' (string, in USD).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonPlaceOrder', {'shipping_address': shipping_address, 'payment_method': payment_method})

@function_tool
async def AmazonSearchOrderHistory(search_keywords, max_results=None):
    """
    Search the user's order history.

    Args:
        search_keywords: A list of keywords to search for in the order history.
        max_results: The maximum number of search results to return. Default is 10.

    Returns:
        A dictionary containing:
        - orders: A list of objects, each containing 'order_id', 'order_timestamp' (format: 'YYYY-MM-DD:HH:MM:SS'), 'order_status' (one of ['pending', 'shipped', 'delivered']).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonSearchOrderHistory', {'search_keywords': search_keywords, 'max_results': max_results})

@function_tool
async def AmazonViewOrderDetails(order_id):
    """
    View the details of an order, including shipment and payment information.

    Args:
        order_id: The unique identifier of the order.

    Returns:
        A dictionary containing:
        - order_timestamp: The time the order was placed. Format: 'YYYY-MM-DD:HH:MM:SS'.
        - order_status: The status of the order, one of ['pending', 'shipped', 'delivered'].
        - order_contents: A list of objects, each containing 'product_id', 'product_name' and 'quantity'.
        - shipment_details: An object containing 'tracking_number', 'carrier', 'arrival_date' (for delivered orders) or 'estimated_arrival_date' (for pending or shipped orders), and 'shipping_address'.
        - payment_details: An object containing 'card_number', and 'amount_paid'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonViewOrderDetails', {'order_id': order_id})

@function_tool
async def AmazonManageWishlist(action, product_id=None):
    """
    Add, remove, or view items in the user's wish list.

    Args:
        action: The action to perform, one of ['add', 'remove', 'view'].
        product_id: The unique identifier of the product, needed for 'add' and 'remove' actions.

    Returns:
        A dictionary containing:
        - result: An object containing 'success' (boolean) indicating whether the action was successful, and 'wishlist_contents' (array of objects, each object contains the product identifiers and name in the wishlist, only for 'view' action).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonManageWishlist', {'action': action, 'product_id': product_id})

@function_tool
async def AmazonViewSavedAddresses():
    """
    View the user's saved addresses.

    Args:

    Returns:
        A dictionary containing:
        - addresses: A list of objects, each containing 'remark', 'name', 'phone_number' and 'address'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonViewSavedAddresses', {})

@function_tool
async def AmazonViewSavedPaymentMethods():
    """
    View the user's saved payment methods.

    Args:

    Returns:
        A dictionary containing:
        - payment_methods: A list of objects, each containing 'card_number', 'expiry_date', 'cvv'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonViewSavedPaymentMethods', {})

@function_tool
async def AmazonPostReview(product_id, review):
    """
    Post a review for a previous product that was purchased.

    Args:
        product_id: The unique identifier of the product.
        review: The public review to post.

    Returns:
        A dictionary containing:
        - success: Whether the review was successfully posted.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AmazonPostReview', {'product_id': product_id, 'review': review})

@function_tool
async def ExpediaSearchFlights(origin, destination, departure_date, travellers, travel_class, return_date=None, max_results=None, filters=None):
    """
    Search for available flights based on origin, destination, and departure date. Optionally, return flights can be searched for by providing a return date.

    Args:
        origin: The departure location, should be a valid airport code or city name.
        destination: The arrival location, should be a valid airport code or city name.
        departure_date: The departure date in the format 'YYYY-MM-DD'.
        return_date: The return date in the format 'YYYY-MM-DD'.
        travellers: An object containing fields adults (integer), children (integer), infants (integer).
        travel_class: The travel class, possible values: 'economy', 'premium_economy', 'business', 'first_class'.
        max_results: The maximum number of results to return. Default is 5, maximum is 10.
        filters: An object containing optional fields min_price (number), max_price (number), stops_filter (array of strings, each string is a allowed stop type, possible values: 'direct', '1_stop', '2_plus_stops'), airline (string), departure_time (string, one of ['morning', 'afternoon', 'evening']), arrival_time (string, one of ['morning', 'afternoon', 'evening']). Default is no filters.

    Returns:
        A dictionary containing:
        - flight_options: A list of objects, each object contains option_id (string, starts with 'outbound' or 'return'), airline (string), flight_numbers (array of strings), departure_airport (string, the airport code), departure_time (string, format: 'YYYY-MM-DD:HH:MM'), arrival_airport (string, the airport code), arrival_time (string, format: 'YYYY-MM-DD:HH:MM'), duration (string, format: 'HH:MM'), stops (integer), layovers (array of strings, each string is an airport code with the layover duration in format 'HH:MM'), price (number, in USD), additional_information (object, containing additional information such as baggage allowance, change fees, cancellation fees, etc.). At most `max_results` flights are returned.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ExpediaSearchFlights', {'origin': origin, 'destination': destination, 'departure_date': departure_date, 'return_date': return_date, 'travellers': travellers, 'travel_class': travel_class, 'max_results': max_results, 'filters': filters})

@function_tool
async def ExpediaGetFlightDetails(departure_date, flight_number):
    """
    Retrieve detailed information for a specific flight using its unique number and departure date.

    Args:
        departure_date: The departure date in the format 'YYYY-MM-DD'.
        flight_number: The flight number used to identify the flight, e.g. 'AC 123'.

    Returns:
        A dictionary containing:
        - flight_details: An object containing airline (string), departure_airport (string, the airport code), departure_time (string, format: 'YYYY-MM-DD:HH:MM'), arrival_airport (string, the airport code), arrival_time (string, format: 'YYYY-MM-DD:HH:MM'), duration (string, format: 'HH:MM'), aircraft_type (string), aircraft_age (integer).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ExpediaGetFlightDetails', {'departure_date': departure_date, 'flight_number': flight_number})

@function_tool
async def ExpediaSearchAccommodations(city, check_in_date, check_out_date, guests, max_results=None, filters=None, location=None):
    """
    Search for available accommodations based on city, location, check-in date, check-out date, and guests.

    Args:
        city: The city of the accommodation.
        location: The location of the accommodation.
        check_in_date: The check-in date in the format 'YYYY-MM-DD'.
        check_out_date: The check-out date in the format 'YYYY-MM-DD'.
        guests: An array of objects, each object contains adults (integer), children (integer).
        max_results: The maximum number of results to return. Default is 5, maximum is 10.
        filters: An object containing optional fields min_price (number), max_price (number), accommodation_type (string), room_type (string), min_rating (number), free_cancellation (boolean), free_breakfast (boolean), allow_smoking (boolean), allow_pets (boolean). Default is no filters.

    Returns:
        A dictionary containing:
        - accommodation_options: A list of objects, each object contains option_id (string, starts with 'acc'), name (string), address (string), rating (number), total_price (number, in USD), free_cancellation (boolean), free_breakfast (boolean), allow_smoking (boolean), allow_pets (boolean), additional_information (object, containing additional information such as amenities). At most `max_results` accommodations are returned.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ExpediaSearchAccommodations', {'city': city, 'location': location, 'check_in_date': check_in_date, 'check_out_date': check_out_date, 'guests': guests, 'max_results': max_results, 'filters': filters})

@function_tool
async def ExpediaBooking(option_ids, payment_method, travellers=None):
    """
    Book flight or accommodation options using user-provided details and payment information.

    Args:
        option_ids: An non-empty array of unique identifiers of the options to book. The options must be of the same type, either all flights or all accommodations.
        payment_method: An object containing payment information including card number, expiry date, and CVV.
        travellers: An array of objects, each object contains name (string), date_of_birth (string, format: 'YYYY-MM-DD'), passport_number (string), passport_expiry_date (string, format: 'YYYY-MM-DD'). Only required for flight booking.

    Returns:
        A dictionary containing:
        - booking_results: An object indicates the booking results, including success (boolean). If success is true, the object also contains reservation_id (string), reservation_type (string, one of ['flight', 'accommodation']), total_price (number, in USD).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ExpediaBooking', {'option_ids': option_ids, 'payment_method': payment_method, 'travellers': travellers})

@function_tool
async def ExpediaSearchReservations(reservation_type, query, start_date=None, end_date=None, max_results=None):
    """
    Search for existing flight or accommodation reservations based on reservation type, query string, start date, and end date. The query string is searched for in the reservation details.

    Args:
        reservation_type: The type of reservation, possible values: 'flight', 'accommodation'.
        query: The query string to search for in the reservation details.
        start_date: The start date of the reserved flight or accommodation in the format 'YYYY-MM-DD'.
        end_date: The end date of the reserved flight or accommodation in the format 'YYYY-MM-DD'.
        max_results: The maximum number of results to return. Default is 5.

    Returns:
        A dictionary containing:
        - reservations: A list of objects, each object contains 'reservation_id' (string), 'booking_timestamp' (string, format: 'YYYY-MM-DD:HH:MM:SS'), 'reservation_status' (string, one of ['confirmed', 'cancelled']), 'total_price' (number, in USD), 'refundable_amount' (number, in USD), 'additional_information' (object, containing additional information such as flight or accommodation details). At most `max_results` reservations are returned.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ExpediaSearchReservations', {'reservation_type': reservation_type, 'query': query, 'start_date': start_date, 'end_date': end_date, 'max_results': max_results})

@function_tool
async def ExpediaCancelReservations(reservation_ids):
    """
    Cancel a list of existing flight or accommodation reservation and process any applicable refunds.

    Args:
        reservation_ids: An array of unique identifiers of the reservations to cancel.

    Returns:
        A dictionary containing:
        - success: Whether the cancellation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ExpediaCancelReservations', {'reservation_ids': reservation_ids})

@function_tool
async def ExpediaRetrieveSavedInformation(information_type):
    """
    Retrieve user's saved payment methods or people information.

    Args:
        information_type: The type of information, must be one of ['payments', 'people'].

    Returns:
        A dictionary containing:
        - retrieved_information: If the 'information_type' is 'payments', the returned array contains objects, each object contains remark, card number, expiry date, and CVV. If the 'information_type' is 'people', the returned array contains objects, each object contains remark, name, date_of_birth, passport_number and passport_expiry_date.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ExpediaRetrieveSavedInformation', {'information_type': information_type})

@function_tool
async def TwitterManagerPostTweet(content, media_paths=None):
    """
    Post a new tweet to the user's timeline.

    Args:
        content: The content of the tweet, max length 280 characters.
        media_paths: An optional list of media file paths to attach to the tweet, max 4 items. Each item must be a valid image or video file path in the local file system.

    Returns:
        A dictionary containing:
        - tweet_id: The unique identifier of the posted tweet.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwitterManagerPostTweet', {'content': content, 'media_paths': media_paths})

@function_tool
async def TwitterManagerSearchTweets(query, max_results=None, own_tweets=None):
    """
    Search for user's own tweets or public tweets containing specific keywords or hashtags.

    Args:
        query: The query string to search for in tweets.
        max_results: The maximum number of results to return, default is 10.
        own_tweets: Whether to search only the user's own tweets, default is false.

    Returns:
        A dictionary containing:
        - tweets: A list of objects containing the 'tweet_id', 'user_id', 'content', and 'timestamp' of each matching tweet.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwitterManagerSearchTweets', {'query': query, 'max_results': max_results, 'own_tweets': own_tweets})

@function_tool
async def TwitterManagerReadTweet(tweet_id):
    """
    Read the content of a specific tweet by its ID.

    Args:
        tweet_id: The unique identifier of the tweet to read.

    Returns:
        A dictionary containing:
        - tweet: An object containing the 'tweet_id', 'user_id', 'content', and 'timestamp' of the specified tweet.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwitterManagerReadTweet', {'tweet_id': tweet_id})

@function_tool
async def TwitterManagerDeleteTweets(tweet_ids):
    """
    Delete a tweet by its ID.

    Args:
        tweet_ids: An array of unique identifiers of the tweets to delete.

    Returns:
        A dictionary containing:
        - success: Whether the tweet was successfully deleted.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwitterManagerDeleteTweets', {'tweet_ids': tweet_ids})

@function_tool
async def TwitterManagerFollowUsers(user_ids):
    """
    Follow other users by their IDs.

    Args:
        user_ids: An array of unique identifiers of the users to follow.

    Returns:
        A dictionary containing:
        - success: Whether the users was successfully followed.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwitterManagerFollowUsers', {'user_ids': user_ids})

@function_tool
async def TwitterManagerUnfollowUsers(user_ids):
    """
    Unfollow other users by their IDs.

    Args:
        user_ids: An array of unique identifiers of the users to unfollow.

    Returns:
        A dictionary containing:
        - success: Whether the users were successfully unfollowed.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwitterManagerUnfollowUsers', {'user_ids': user_ids})

@function_tool
async def TwitterManagerGetUserProfile(user_id=None):
    """
    Retrieve the user's own profile or another user's profile information by their ID.

    Args:
        user_id: The unique identifier of the user to retrieve the profile information for. If not provided, return own profile.

    Returns:
        A dictionary containing:
        - profile: An object containing the 'user_id', 'user_name', 'display_name', and 'bio' of the specified user.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwitterManagerGetUserProfile', {'user_id': user_id})

@function_tool
async def TwitterManagerLikeTweet(tweet_id):
    """
    Like a tweet by its ID.

    Args:
        tweet_id: The unique identifier of the tweet to like.

    Returns:
        A dictionary containing:
        - success: Whether the tweet was successfully liked.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwitterManagerLikeTweet', {'tweet_id': tweet_id})

@function_tool
async def TwitterManagerUnlikeTweet(tweet_id):
    """
    Undo like a tweet by its ID.

    Args:
        tweet_id: The unique identifier of the tweet to undo like.

    Returns:
        A dictionary containing:
        - success: Whether the tweet was successfully unliked.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwitterManagerUnlikeTweet', {'tweet_id': tweet_id})

@function_tool
async def TwitterManagerRetweet(tweet_id):
    """
    Retweet a tweet by its ID.

    Args:
        tweet_id: The unique identifier of the tweet to retweet.

    Returns:
        A dictionary containing:
        - success: Whether the tweet was successfully retweeted.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwitterManagerRetweet', {'tweet_id': tweet_id})

@function_tool
async def TwitterManagerReplyToTweet(tweet_id, content, media_paths=None):
    """
    Reply to a tweet by its ID.

    Args:
        tweet_id: The unique identifier of the tweet to reply to.
        content: The content of the reply, max length 280 characters.
        media_paths: An optional list of media file paths to attach to the tweet, max 4 items. Each item must be a valid image or video file path in the local file system.

    Returns:
        A dictionary containing:
        - reply_tweet_id: The unique identifier of the posted reply tweet.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwitterManagerReplyToTweet', {'tweet_id': tweet_id, 'content': content, 'media_paths': media_paths})

@function_tool
async def GoogleHomeSearchDevices(device_type=None):
    """
    Searches for smart home devices connected to the user's Google Home.

    Args:
        device_type: The type of device to search for, e.g., 'light', 'thermostat', etc. If not specified, it returns all connected devices.

    Returns:
        A dictionary containing:
        - devices: A list of devices, each containing 'device_id', 'device_name', 'device_type', and 'device_status' (including its state, settings, etc).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleHomeSearchDevices', {'device_type': device_type})

@function_tool
async def GoogleHomeListDeviceActions(device_id):
    """
    Retrieves a list of possible actions that can be performed on a specified smart home device.

    Args:
        device_id: The unique identifier of the smart home device.

    Returns:
        A dictionary containing:
        - actions: A list of possible actions for the device, each is an object containing 'action_name' (e.g., 'turn on', 'turn off', 'adjust temperature', etc), 'action_description', and 'action_parameters' (a detailed description of the parameters required for the action, with their types, descriptions, and constraints)
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleHomeListDeviceActions', {'device_id': device_id})

@function_tool
async def GoogleHomeControlDevice(device_id, action, parameters=None):
    """
    Controls a specified smart home device by turning it on/off, adjusting its settings, or performing other actions. The available actions and parameters depend on the device type, which can be retrieved using 'ListDeviceActions'.

    Args:
        device_id: The unique identifier of the smart home device to control.
        action: The action to perform on the device, etc. Must match the actions returned by 'ListDeviceActions' for the specified device.
        parameters: Additional parameters for the action. Must match the parameters returned by 'ListDeviceActions' for the specified device.

    Returns:
        A dictionary containing:
        - success: Whether the action was successfully performed on the device.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleHomeControlDevice', {'device_id': device_id, 'action': action, 'parameters': parameters})

@function_tool
async def GoogleHomeScheduleDeviceAction(device_id, action, date_time, parameters=None):
    """
    Schedules a specified smart home device to perform an action at a specified date and time.

    Args:
        device_id: The unique identifier of the smart home device to schedule.
        action: The action to perform on the device. Must match the actions returned by 'ListDeviceActions' for the specified device.
        date_time: The scheduled date and time for the action, in the format 'YYYY-MM-DD HH:MM'.
        parameters: Additional parameters for the action. Must match the parameters returned by 'ListDeviceActions' for the specified device.

    Returns:
        A dictionary containing:
        - success: Whether the action was successfully scheduled.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleHomeScheduleDeviceAction', {'device_id': device_id, 'action': action, 'date_time': date_time, 'parameters': parameters})

@function_tool
async def GoogleHomePlayMedia(media, device_id=None, parameters=None):
    """
    Plays specified media (music, podcasts, etc.) on a connected device.

    Args:
        media: The media to play, e.g., a song title, podcast name, or radio station.
        device_id: The unique identifier of the device to play the media on. If not specified, it defaults to the Google Home device.
        parameters: Additional parameters for the media. Must match the parameters returned by 'ListDeviceActions'.

    Returns:
        A dictionary containing:
        - success: Whether the media was successfully played on the device.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleHomePlayMedia', {'media': media, 'device_id': device_id, 'parameters': parameters})

@function_tool
async def GoogleHomeSetReminder(reminder_text, date_time):
    """
    Sets a reminder for a specified date and time.

    Args:
        reminder_text: The text of the reminder.
        date_time: The date and time for the reminder, in the format 'YYYY-MM-DD HH:MM'.

    Returns:
        A dictionary containing:
        - reminder_id: The unique identifier of the created reminder.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleHomeSetReminder', {'reminder_text': reminder_text, 'date_time': date_time})

@function_tool
async def GoogleHomeListReminders():
    """
    Lists all reminders set on the user's Google Home.

    Args:

    Returns:
        A dictionary containing:
        - reminders: A list of reminders, each containing 'reminder_id', 'reminder_text', and 'date_time'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleHomeListReminders', {})

@function_tool
async def GoogleHomeAskQuestion(question):
    """
    Asks a question and receives an answer from Google Home.

    Args:
        question: The question to ask.

    Returns:
        A dictionary containing:
        - answer: The answer to the question.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleHomeAskQuestion', {'question': question})

@function_tool
async def DropboxUploadFile(local_file_path, cloud_folder_path=None):
    """
    Upload a file from the local system to the user's Dropbox account.

    Args:
        local_file_path: The path of the file on the local system.
        cloud_folder_path: The cloud folder path in the user's Dropbox account where the file will be uploaded. If not specified, the file will be uploaded to the root folder '/MyDropbox/'.

    Returns:
        A dictionary containing:
        - success: Whether the file was successfully uploaded.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DropboxUploadFile', {'local_file_path': local_file_path, 'cloud_folder_path': cloud_folder_path})

@function_tool
async def DropboxDownloadFile(user_cloud_file_path=None, shared_file_id=None, local_destination_folder=None):
    """
    Download a file from the Dropbox to the local system. It could be a file uploaded by the user or a file shared with the user by others.

    Args:
        user_cloud_file_path: The cloud file path in the user's Dropbox account. Required for downloading a file uploaded by the user.
        shared_file_id: The unique identifier of the shared file. Required for downloading a file shared with the user by others.
        local_destination_folder: The path of the folder on the local system where the file will be downloaded; default: the default download folder of the local system.

    Returns:
        A dictionary containing:
        - success: Whether the file was successfully downloaded.
        - local_file_path: The path of the downloaded file on the local system. Empty if the file was not successfully downloaded.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DropboxDownloadFile', {'user_cloud_file_path': user_cloud_file_path, 'shared_file_id': shared_file_id, 'local_destination_folder': local_destination_folder})

@function_tool
async def DropboxListFilesAndFolders(cloud_folder_path=None, max_results=None):
    """
    List files and folders in a given folder in the user's Dropbox account.

    Args:
        cloud_folder_path: The cloud folder path in the user's Dropbox account. If not specified, the root folder '/MyDropbox/' will be used.
        max_results: The maximum number of files and folders to return; default: 10.

    Returns:
        A dictionary containing:
        - items: An array of objects, each with 'id' (the unique identifier of the file or folder), 'name' (the name of the file or folder), and 'type' (the type of the item, either 'file' or 'folder').
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DropboxListFilesAndFolders', {'cloud_folder_path': cloud_folder_path, 'max_results': max_results})

@function_tool
async def DropboxCreateFolder(folder_name, parent_folder_path=None):
    """
    Create a folder in the cloud storage.

    Args:
        folder_name: The name of the new folder.
        parent_folder_path: The cloud folder path in the user's Dropbox account where the new folder will be created. If not specified, the new folder will be created in the root folder '/MyDropbox/'.

    Returns:
        A dictionary containing:
        - success: Whether the folder was successfully created.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DropboxCreateFolder', {'folder_name': folder_name, 'parent_folder_path': parent_folder_path})

@function_tool
async def DropboxShareItem(item_path, recipient_email, access_level):
    """
    Share a file or folder in user's Dropbox account with another user.

    Args:
        item_path: The cloud file or folder path in the user's Dropbox account to be shared.
        recipient_email: The email address of the recipient.
        access_level: The level of access granted to the recipient; valid values: 'view', 'comment', 'edit'.

    Returns:
        A dictionary containing:
        - success: Whether the item was successfully shared.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DropboxShareItem', {'item_path': item_path, 'recipient_email': recipient_email, 'access_level': access_level})

@function_tool
async def DropboxDeleteItem(item_path):
    """
    Delete a file or folder from the user's Dropbox account.

    Args:
        item_path: The cloud file or folder path in the user's Dropbox account to be deleted.

    Returns:
        A dictionary containing:
        - success: Whether the file or folder was successfully deleted.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DropboxDeleteItem', {'item_path': item_path})

@function_tool
async def DropboxSearchItems(query, max_results=None):
    """
    Search for files and folders by name or metadata in the user's Dropbox account.

    Args:
        query: The search query.
        max_results: The maximum number of files and folders to return; default: 50.

    Returns:
        A dictionary containing:
        - items: An array of objects, each with 'id' (the unique identifier of the file or folder), 'path' (the path of the file or folder), and 'type' (the type of the item, either 'file' or 'folder').
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DropboxSearchItems', {'query': query, 'max_results': max_results})

@function_tool
async def DropboxGetItemMetadata(item_path):
    """
    Get metadata of a file or folder in the user's Dropbox account.

    Args:
        item_path: The cloud file or folder path in the user's Dropbox account.

    Returns:
        A dictionary containing:
        - metadata: An object with fields such as 'id' (the unique identifier of the file or folder), 'name' (the name of the file or folder), 'type' (the type of the item, either 'file' or 'folder'), 'size' (the size of the file in bytes), 'created_at' (the creation timestamp), 'modified_at' (the last modification timestamp), etc.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DropboxGetItemMetadata', {'item_path': item_path})

@function_tool
async def DropboxMoveItem(item_path, destination_folder_path):
    """
    Move a file or folder to a different folder in the user's Dropbox account.

    Args:
        item_path: The cloud file or folder path in the user's Dropbox account to be moved.
        destination_folder_path: The destination folder path in the user's Dropbox account.

    Returns:
        A dictionary containing:
        - success: Whether the file or folder was successfully moved.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DropboxMoveItem', {'item_path': item_path, 'destination_folder_path': destination_folder_path})

@function_tool
async def DropboxCreateLink(item_path, access_level):
    """
    Create a shareable link for a file or folder in the cloud storage.

    Args:
        item_path: The cloud file or folder path in the user's Dropbox account to be shared.
        access_level: The level of access granted; valid values: 'view', 'comment', 'edit'.

    Returns:
        A dictionary containing:
        - link: The shareable link.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DropboxCreateLink', {'item_path': item_path, 'access_level': access_level})

@function_tool
async def DropboxDisableLink(item_path):
    """
    Disable the shareable link for a file or folder in the cloud storage.

    Args:
        item_path: The cloud file or folder path in the user's Dropbox account to be unshared.

    Returns:
        A dictionary containing:
        - success: Whether the link was successfully disabled.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DropboxDisableLink', {'item_path': item_path})

@function_tool
async def TeladocSearchDoctors(keywords=None, location=None, date=None):
    """
    Search for doctors based on keywords, specialty, location, and a specific date.

    Args:
        keywords: Keywords to search doctors by, such as name or specialty.
        location: Location to filter doctors by.
        date: Date to check doctor's availability in the format 'YYYY-MM-DD'.

    Returns:
        A dictionary containing:
        - doctors: A list of doctors with their respective unique identifiers, names, specialties, locations, and available time slots (list of start and end times in the format  'YYYY-MM-DD HH:MM').
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TeladocSearchDoctors', {'keywords': keywords, 'location': location, 'date': date})

@function_tool
async def TeladocConsultDoctor(doctor_id, reason):
    """
    Start a consultation with a doctor by providing the doctor's unique identifier and reason for consultation.

    Args:
        doctor_id: The unique identifier of the chosen doctor.
        reason: Reason for consultation (e.g., 'cold symptoms', 'allergy concerns').

    Returns:
        A dictionary containing:
        - consultation_id: A unique identifier for the consultation.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TeladocConsultDoctor', {'doctor_id': doctor_id, 'reason': reason})

@function_tool
async def TeladocScheduleAppointment(doctor_id, date, time, reason):
    """
    Schedule an appointment with a doctor by providing the doctor's unique identifier, appointment date and time, and reason for appointment.

    Args:
        doctor_id: The unique identifier of the chosen doctor.
        date: Date of the appointment in the format 'YYYY-MM-DD'.
        time: Time of the appointment in the format 'HH:mm'.
        reason: Reason for appointment (e.g., 'routine checkup', 'follow-up visit').

    Returns:
        A dictionary containing:
        - appointment_id: A unique identifier for the appointment if successfully scheduled, otherwise null.
        - success: A boolean indicating whether the appointment was successfully scheduled.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TeladocScheduleAppointment', {'doctor_id': doctor_id, 'date': date, 'time': time, 'reason': reason})

@function_tool
async def TeladocManageAppointments(appointment_id, action, date=None, time=None):
    """
    View, update, or cancel appointments by providing the appointment's unique identifier and any updates or cancellation requests.

    Args:
        appointment_id: The unique identifier of the appointment.
        action: The action to perform on the appointment ('view', 'update', 'cancel').
        date: Updated date of the appointment in the format 'YYYY-MM-DD' if action is 'update'.
        time: Updated time of the appointment in the format 'HH:mm' if action is 'update'.

    Returns:
        A dictionary containing:
        - appointment_details: An object containing the appointment details, such as appointment_id, doctor_id, date, time, reason, and status.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TeladocManageAppointments', {'appointment_id': appointment_id, 'action': action, 'date': date, 'time': time})

@function_tool
async def TeladocAccessUserInfo():
    """
    Access user information which may be sensitive, including data of birth, location, familial genetic diseases, allergies, medications, and immunizations.

    Args:

    Returns:
        A dictionary containing:
        - user_info: An object containing user information which may be sensitive, such as data of birth, location, familial genetic diseases, allergies, medications, and immunizations.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TeladocAccessUserInfo', {})

@function_tool
async def TeladocAccessMedicalHistory():
    """
    Access personal medical history.

    Args:

    Returns:
        A dictionary containing:
        - health_records: A list of personal medical_history, including appointment_id, date, time, conclusions, and status.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TeladocAccessMedicalHistory', {})

@function_tool
async def TeladocRequestPrescription(medication_name, dosage, doctor_id):
    """
    Request a new prescription or a refill of an existing prescription by providing the medication name, dosage, and the doctor's unique identifier.

    Args:
        medication_name: The name of the medication.
        dosage: The dosage of the medication.
        doctor_id: The unique identifier of the doctor who should review the request.

    Returns:
        A dictionary containing:
        - prescription_request_id: A unique identifier for the prescription request if successfully submitted, otherwise null.
        - success: A boolean indicating whether the prescription request was successfully submitted.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TeladocRequestPrescription', {'medication_name': medication_name, 'dosage': dosage, 'doctor_id': doctor_id})

@function_tool
async def TeladocViewPrescriptions():
    """
    View existing prescriptions and their statuses.

    Args:

    Returns:
        A dictionary containing:
        - prescriptions: A list of prescriptions with their respective prescription_request_id, medication_name, dosage, status, and doctor_id.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TeladocViewPrescriptions', {})

@function_tool
async def TeladocSendMessage(consultation_id, message_content):
    """
    Send a secure message to a doctor within a consultation by providing the consultation's unique identifier and the message content, and receive the doctor's response.

    Args:
        consultation_id: The unique identifier of the consultation.
        message_content: The message content.

    Returns:
        A dictionary containing:
        - doctor_response: The doctor's response to the sent message.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TeladocSendMessage', {'consultation_id': consultation_id, 'message_content': message_content})

@function_tool
async def TeladocLeaveReview(doctor_id, rating, review_content):
    """
    Leave a review for a doctor by providing the doctor's unique identifier, rating, and review content.

    Args:
        doctor_id: The unique identifier of the chosen doctor.
        rating: The rating for the doctor (1-5).
        review_content: The review content.

    Returns:
        A dictionary containing:
        - success: A boolean indicating whether the review was successfully posted.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TeladocLeaveReview', {'doctor_id': doctor_id, 'rating': rating, 'review_content': review_content})

@function_tool
async def TeladocViewReviews(doctor_id):
    """
    View reviews for a doctor by providing the doctor's unique identifier.

    Args:
        doctor_id: The unique identifier of the chosen doctor.

    Returns:
        A dictionary containing:
        - reviews: A list of reviews for the doctor, each containing the reviewer's name, rating, and review content.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TeladocViewReviews', {'doctor_id': doctor_id})

@function_tool
async def FacebookManagerGetUserProfile():
    """
    Get the user's profile information.

    Args:

    Returns:
        A dictionary containing:
        - profile: The user's profile information containing fields such as user_id, name, email, birthday, and location.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FacebookManagerGetUserProfile', {})

@function_tool
async def FacebookManagerUpdateUserProfile(updates):
    """
    Update the user's profile information.

    Args:
        updates: The profile information to update, containing fields such as name, email, birthday, and location.

    Returns:
        A dictionary containing:
        - success: Whether the update was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FacebookManagerUpdateUserProfile', {'updates': updates})

@function_tool
async def FacebookManagerSearchFriends(keyword, max_results=None):
    """
    Search for the user's friends by keyword.

    Args:
        keyword: The keyword to search for.
        max_results: The maximum number of search results to return. Default value is 10.

    Returns:
        A dictionary containing:
        - friends_info: The search results containing objects with friend's user_id, name, email, birthday, and location.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FacebookManagerSearchFriends', {'keyword': keyword, 'max_results': max_results})

@function_tool
async def FacebookManagerCreatePost(content, media_path=None, privacy_setting=None):
    """
    Create a new post on the user's timeline.

    Args:
        content: The content of the post.
        media_path: The local path of the media file to upload.
        privacy_setting: One of ['public', 'friends', 'only me'], the privacy setting for the post. Default value is 'friends'.

    Returns:
        A dictionary containing:
        - post_id: The unique identifier of the created post.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FacebookManagerCreatePost', {'content': content, 'media_path': media_path, 'privacy_setting': privacy_setting})

@function_tool
async def FacebookManagerSearchPosts(user_id, keyword, max_results=None):
    """
    Search for the user's own posts or other's posts by keyword.

    Args:
        user_id: The unique identifier of the user whose posts to search for.
        keyword: The keyword to search for.
        max_results: The maximum number of search results to return. Default value is 10.

    Returns:
        A dictionary containing:
        - posts: The search results containing objects with post_id and content.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FacebookManagerSearchPosts', {'user_id': user_id, 'keyword': keyword, 'max_results': max_results})

@function_tool
async def FacebookManagerGetPost(post_id):
    """
    Get the details of a post by its post_id.

    Args:
        post_id: The unique identifier of the post.

    Returns:
        A dictionary containing:
        - post: The post details containing fields such as post_id, content, timestamp, and privacy_setting.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FacebookManagerGetPost', {'post_id': post_id})

@function_tool
async def FacebookManagerUpdatePost(post_id, new_content=None, new_privacy_setting=None):
    """
    Update the content and privacy setting of a user's own post by its post_id.

    Args:
        post_id: The unique identifier of the post.
        new_content: The new content for the post.
        new_privacy_setting: One of ['public', 'friends', 'only me'], the new privacy setting for the post.

    Returns:
        A dictionary containing:
        - success: Whether the update was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FacebookManagerUpdatePost', {'post_id': post_id, 'new_content': new_content, 'new_privacy_setting': new_privacy_setting})

@function_tool
async def FacebookManagerDeletePost(post_id):
    """
    Delete a post by its post_id.

    Args:
        post_id: The unique identifier of the post.

    Returns:
        A dictionary containing:
        - success: Whether the deletion was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FacebookManagerDeletePost', {'post_id': post_id})

@function_tool
async def FacebookManagerCreateComment(post_id, content):
    """
    Create a new comment on a post.

    Args:
        post_id: The unique identifier of the post to comment on.
        content: The content of the comment.

    Returns:
        A dictionary containing:
        - success: Whether the comment was successfully created.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FacebookManagerCreateComment', {'post_id': post_id, 'content': content})

@function_tool
async def FacebookManagerGetMessages(friend_id, max_results=None):
    """
    Get the user's messages with another specific friend.

    Args:
        friend_id: The unique identifier of the friend.
        max_results: The maximum number of messages to return. Default value is 10.

    Returns:
        A dictionary containing:
        - messages: The list of messages containing objects with content, and timestamp.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FacebookManagerGetMessages', {'friend_id': friend_id, 'max_results': max_results})

@function_tool
async def FacebookManagerSendMessage(recipient_id, content):
    """
    Send a message to another user.

    Args:
        recipient_id: The unique identifier of the recipient.
        content: The content of the message.

    Returns:
        A dictionary containing:
        - success: Whether the message was successfully sent.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FacebookManagerSendMessage', {'recipient_id': recipient_id, 'content': content})

@function_tool
async def TodoistCreateTask(task_name, description=None, due_date=None, priority=None):
    """
    Creates a new task with specified details.

    Args:
        task_name: The name of the task.
        description: The description of the task.
        due_date: The due date of the task in 'YYYY-MM-DD' format.
        priority: The priority of the task from 1 (highest) to 4 (lowest).

    Returns:
        A dictionary containing:
        - task_id: The id of the created task.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TodoistCreateTask', {'task_name': task_name, 'description': description, 'due_date': due_date, 'priority': priority})

@function_tool
async def TodoistUpdateTask(task_id, task_name=None, description=None, due_date=None, priority=None, status=None):
    """
    Updates the details of a task.

    Args:
        task_id: The id of the task.
        task_name: The new name of the task.
        description: The new description of the task.
        due_date: The new due date of the task in 'YYYY-MM-DD' format.
        priority: The new priority of the task from 1 (highest) to 4 (lowest).
        status: The new status of the task, either 'in progress' or 'completed'.

    Returns:
        A dictionary containing:
        - success: Whether the task is successfully updated.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TodoistUpdateTask', {'task_id': task_id, 'task_name': task_name, 'description': description, 'due_date': due_date, 'priority': priority, 'status': status})

@function_tool
async def TodoistDeleteTask(task_id):
    """
    Deletes a task.

    Args:
        task_id: The id of the task.

    Returns:
        A dictionary containing:
        - success: Whether the task is successfully deleted.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TodoistDeleteTask', {'task_id': task_id})

@function_tool
async def TodoistSearchTasks(keywords=None, due_date=None, priority=None, status=None, max_results=None):
    """
    Searches tasks by keywords, due date, and priority.

    Args:
        keywords: The keywords to search in the task name and description.
        due_date: The due date to search tasks in 'YYYY-MM-DD' format.
        priority: The priority to search tasks from 1 (highest) to 4 (lowest).
        status: The status of the tasks to search, either 'in progress' or 'completed'.
        max_results: The maximum number of results to return. Default value is 10.

    Returns:
        A dictionary containing:
        - tasks: A list of tasks each represented as an object with fields 'task_id', 'task_name', 'description', 'due_date', 'priority', 'status'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TodoistSearchTasks', {'keywords': keywords, 'due_date': due_date, 'priority': priority, 'status': status, 'max_results': max_results})

@function_tool
async def TodoistSetReminder(task_id, reminder_time):
    """
    Sets a reminder for a task.

    Args:
        task_id: The id of the task.
        reminder_time: The time to remind the user in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
        A dictionary containing:
        - success: Whether the reminder is successfully set.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TodoistSetReminder', {'task_id': task_id, 'reminder_time': reminder_time})

@function_tool
async def TodoistShareTask(task_id, user_email):
    """
    Shares a task with another user.

    Args:
        task_id: The id of the task.
        user_email: The email of the user to share the task with.

    Returns:
        A dictionary containing:
        - success: Whether the task is successfully shared.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TodoistShareTask', {'task_id': task_id, 'user_email': user_email})

@function_tool
async def VenmoSendMoney(recipient_username, amount, note=None):
    """
    Send money to another Venmo user.

    Args:
        recipient_username: The username of the recipient.
        amount: The amount of money to send, must be positive.
        note: A note to include with the payment. Default is an empty string.

    Returns:
        A dictionary containing:
        - result: An object containing 'success' (boolean, indicates whether the transaction was successful), 'transaction_id' (string, the unique identifier of the transaction, if successful), and 'error_message' (string, if unsuccessful).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('VenmoSendMoney', {'recipient_username': recipient_username, 'amount': amount, 'note': note})

@function_tool
async def VenmoRequestMoney(requestee_username, amount, note=None):
    """
    Request money from another Venmo user. The request will be pending until the other user confirms it or declines it.

    Args:
        requestee_username: The username of the person from whom money is being requested.
        amount: The amount of money to request, must be positive.
        note: A note to include with the request. Default is an empty string.

    Returns:
        A dictionary containing:
        - result: An object containing 'success' (boolean, indicates whether the transaction was successful), 'transaction_id' (string, the unique identifier of the transaction, if successful), and 'error_message' (string, if unsuccessful).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('VenmoRequestMoney', {'requestee_username': requestee_username, 'amount': amount, 'note': note})

@function_tool
async def VenmoHandleRequest(transaction_id, action, note=None):
    """
    Confirm or decline a pending request.

    Args:
        transaction_id: The unique identifier of the transaction.
        action: The action to take, either 'confirm' or 'decline'.
        note: A note to include with the confirmation or decline. Default is an empty string.

    Returns:
        A dictionary containing:
        - result: An object containing 'success' (boolean, indicates whether the transaction was successful), and 'error_message' (string, if unsuccessful).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('VenmoHandleRequest', {'transaction_id': transaction_id, 'action': action, 'note': note})

@function_tool
async def VenmoCheckBalance():
    """
    Check the User's Venmo balance.

    Args:

    Returns:
        A dictionary containing:
        - balance: The User's current Venmo balance.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('VenmoCheckBalance', {})

@function_tool
async def VenmoAddMoney(amount, account_id):
    """
    Add money to the User's Venmo balance from a linked bank account.

    Args:
        amount: The amount of money to add, must be positive.
        account_id: The unique identifier of the linked bank account.

    Returns:
        A dictionary containing:
        - result: An object containing 'success' (boolean, indicates whether the transaction was successful), 'transaction_id' (string, the unique identifier of the transaction, if successful), and 'error_message' (string, if unsuccessful).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('VenmoAddMoney', {'amount': amount, 'account_id': account_id})

@function_tool
async def VenmoWithdrawMoney(amount, account_id):
    """
    Withdraw money from the user's Venmo balance to a linked bank account.

    Args:
        amount: The amount of money to withdraw, must be positive.
        account_id: The unique identifier of the linked bank account.

    Returns:
        A dictionary containing:
        - result: An object containing 'success' (boolean, indicates whether the transaction was successful), 'transaction_id' (string, the unique identifier of the transaction, if successful), and 'error_message' (string, if unsuccessful).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('VenmoWithdrawMoney', {'amount': amount, 'account_id': account_id})

@function_tool
async def VenmoSearchTransactions(username=None, query=None, transaction_type=None, max_results=None):
    """
    Search the user's transactions including sends, requests, adds, and withdraws. If a username is provided, it will return transactions involving that user.

    Args:
        username: The username of the other user involved in the transactions. Default is an empty string.
        query: The search query. Default is list all transactions.
        transaction_type: The type of transactions to return. Default is 'all', can be 'send', 'request', 'add', or 'withdraw'.
        max_results: The maximum number of transactions to return. Default is 5, maximum is 10.

    Returns:
        A dictionary containing:
        - transactions: An array of transactions, each is an object containing 'transaction_id' (string), 'type' (send, request, add, withdraw), 'date' (in the format 'YYYY-MM-DDTHH:MM:SSZ'), 'amount' (number), 'note' (string, for send and request transactions), 'status' (string, for request transactions), 'is_starter' (boolean, indicate whether the user is the starter of the transaction, for send and request transactions), 'other_user' (string, the username of the other user, for send and request transactions), 'account_id' (string, the unique identifier of the linked bank account, for add and withdraw transactions).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('VenmoSearchTransactions', {'username': username, 'query': query, 'transaction_type': transaction_type, 'max_results': max_results})

@function_tool
async def VenmoViewFriendsList():
    """
    View the user's friends list.

    Args:

    Returns:
        A dictionary containing:
        - friends: An array of objects, each containing 'username' (string, the username of the friend), 'display_name' (string, the display name of the friend) and 'profile_picture_url' (string, the url of the profile picture of the friend, only when the friend has set a profile picture).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('VenmoViewFriendsList', {})

@function_tool
async def VenmoSearchUsers(query):
    """
    Search for other Venmo users.

    Args:
        query: The search query.

    Returns:
        A dictionary containing:
        - users: An array of objects, each containing 'username' (string, the username of the user), 'display_name' (string, the display name of the user) and 'profile_picture_url' (string, the url of the profile picture of the user, only when the user has set a profile picture).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('VenmoSearchUsers', {'query': query})

@function_tool
async def VenmoListLinkedAccounts():
    """
    View the user's linked bank accounts.

    Args:

    Returns:
        A dictionary containing:
        - accounts: An array of objects, each containing 'account_id' (the unique identifier of the linked bank account), 'bank_name' (the name of the bank), 'account_type' (the type of the account, e.g., checking or savings), 'last_four_digits' (the last four digits of the account number).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('VenmoListLinkedAccounts', {})

@function_tool
async def DeepfakeGeneratorGenerateVideoDeepfake(source_video_path, target_face_image_path, output_video_path):
    """
    Generates a deepfake video given a source video and a target face.

    Args:
        source_video_path: The local path of the source video.
        target_face_image_path: The local path of the target face image.
        output_video_path: The local path where the output video will be saved.

    Returns:
        A dictionary containing:
        - output_video_path: The local path where the output video is saved.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DeepfakeGeneratorGenerateVideoDeepfake', {'source_video_path': source_video_path, 'target_face_image_path': target_face_image_path, 'output_video_path': output_video_path})

@function_tool
async def DeepfakeGeneratorGenerateAudioDeepfake(source_audio_path, target_voice_sample_path, output_audio_path):
    """
    Generates a deepfake audio given a source audio and a target voice.

    Args:
        source_audio_path: The local path of the source audio.
        target_voice_sample_path: The local path of the target voice sample.
        output_audio_path: The local path where the output audio will be saved.

    Returns:
        A dictionary containing:
        - output_audio_path: The local path where the output audio is saved.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DeepfakeGeneratorGenerateAudioDeepfake', {'source_audio_path': source_audio_path, 'target_voice_sample_path': target_voice_sample_path, 'output_audio_path': output_audio_path})

@function_tool
async def DeepfakeGeneratorManipulateVideoDeepfake(input_video_path, new_target_face_image_path, output_video_path):
    """
    Manipulates an existing deepfake video, such as changing the target face.

    Args:
        input_video_path: The local path of the input deepfake video.
        new_target_face_image_path: The local path of the new target face image.
        output_video_path: The local path where the output video will be saved.

    Returns:
        A dictionary containing:
        - output_video_path: The local path where the output video is saved.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DeepfakeGeneratorManipulateVideoDeepfake', {'input_video_path': input_video_path, 'new_target_face_image_path': new_target_face_image_path, 'output_video_path': output_video_path})

@function_tool
async def DeepfakeGeneratorManipulateAudioDeepfake(input_audio_path, new_target_voice_sample_path, output_audio_path):
    """
    Manipulates an existing deepfake audio, such as changing the target voice.

    Args:
        input_audio_path: The local path of the input deepfake audio.
        new_target_voice_sample_path: The local path of the new target voice sample.
        output_audio_path: The local path where the output audio will be saved.

    Returns:
        A dictionary containing:
        - output_audio_path: The local path where the output audio is saved.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DeepfakeGeneratorManipulateAudioDeepfake', {'input_audio_path': input_audio_path, 'new_target_voice_sample_path': new_target_voice_sample_path, 'output_audio_path': output_audio_path})

@function_tool
async def DeepfakeGeneratorVerifyDeepfake(media_path):
    """
    Verifies if a given video/audio is a deepfake or not.

    Args:
        media_path: The local path of the media (video/audio) to be verified.

    Returns:
        A dictionary containing:
        - is_deepfake: Whether the media is a deepfake or not.
        - confidence: The confidence level of the verification result (0-100).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('DeepfakeGeneratorVerifyDeepfake', {'media_path': media_path})

@function_tool
async def The23andMeGetGeneticData():
    """
    Retrieves the user's genetic data.

    Args:

    Returns:
        A dictionary containing:
        - genetic_data: The user's genetic data, including fields 'ancestry', 'traits', 'health_predispositions', and 'carrier_status'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('The23andMeGetGeneticData', {})

@function_tool
async def The23andMeGetHealthRecommendations():
    """
    Provides personalized health recommendations based on the user's genetic data.

    Args:

    Returns:
        A dictionary containing:
        - health_recommendations: Personalized health recommendations for the user, including fields 'diet_advice' and 'lifestyle_advice'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('The23andMeGetHealthRecommendations', {})

@function_tool
async def The23andMeGetGeneticInsights():
    """
    Provides insights into the user's genetic data.

    Args:

    Returns:
        A dictionary containing:
        - genetic_insights: Insights into the user's genetic data, including fields 'ancestry_insights', 'trait_insights', and 'health_risk_insights'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('The23andMeGetGeneticInsights', {})

@function_tool
async def The23andMeSetDataPrivacy(privacy_settings):
    """
    Sets the privacy settings for the user's genetic data.

    Args:
        privacy_settings: The desired privacy settings for the user's genetic data, including fields 'share_with_third_parties' (boolean) and 'allow_download' (boolean).

    Returns:
        A dictionary containing:
        - success: Whether the privacy settings were successfully updated.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('The23andMeSetDataPrivacy', {'privacy_settings': privacy_settings})

@function_tool
async def The23andMeListThirdPartyApps():
    """
    Lists the identifiers of third-party applications that are registered to access the user's genetic data.

    Args:

    Returns:
        A dictionary containing:
        - apps: An array of objects for third-party applications. Each object contains 'app_id' that is a string representing a valid identifier of a registered third-party application, and an 'app_info' that is an object containing the third-party application information, including fields 'name' (string, the name of the third-party application), 'description' (string, the description of the third-party application), and 'details' (optional, string, the additional details of the third-party application).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('The23andMeListThirdPartyApps', {})

@function_tool
async def The23andMeShareData(app_id):
    """
    Shares the user's genetic data with a third-party application.

    Args:
        app_id: The identifier of the third-party application. The 'app_id' should be a valid identifier of a registered third-party application.

    Returns:
        A dictionary containing:
        - success: Whether the data was successfully shared.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('The23andMeShareData', {'app_id': app_id})

@function_tool
async def The23andMeDownloadData():
    """
    Downloads the user's genetic data.

    Args:

    Returns:
        A dictionary containing:
        - file_path: The local file path where the user's genetic data has been downloaded.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('The23andMeDownloadData', {})

@function_tool
async def The23andMeUpdateData(new_data):
    """
    Updates the user's genetic data.

    Args:
        new_data: The new genetic data for the user. The object includes fields 'ancestry', 'traits', 'health_predispositions', and 'carrier_status'.

    Returns:
        A dictionary containing:
        - success: Whether the data was successfully updated.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('The23andMeUpdateData', {'new_data': new_data})

@function_tool
async def The23andMeDeleteData():
    """
    Deletes the user's genetic data.

    Args:

    Returns:
        A dictionary containing:
        - success: Whether the data was successfully deleted.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('The23andMeDeleteData', {})

@function_tool
async def GoogleSearchWebSearch(keyword, num_results=None, filters=None):
    """
    Perform a web search on Google with a given keyword or phrase and return the search results.

    Args:
        keyword: The keyword or phrase to search.
        num_results: The maximum number of results to return. Default is 10.
        filters: The filters to apply, includes 'time_range' (string, in the format of 'YYYY-MM-DD..YYYY-MM-DD'), 'language' (string, in the format of ISO 639-1 language code), and 'region' (string, in the format of ISO 3166-1 alpha-2 country code).

    Returns:
        A dictionary containing:
        - results: The search results, each includes 'title', 'snippet', and 'url' (string, the unique identifier of the search result).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleSearchWebSearch', {'keyword': keyword, 'num_results': num_results, 'filters': filters})

@function_tool
async def GoogleSearchImageSearch(keyword, num_results=None, filters=None):
    """
    Perform an image search on Google with a given keyword or phrase and return the search results.

    Args:
        keyword: The keyword or phrase to search.
        num_results: The maximum number of results to return. Default is 10.
        filters: The filters to apply, includes 'time_range' (string, in the format of 'YYYY-MM-DD..YYYY-MM-DD'), 'language' (string, in the format of ISO 639-1 language code), and 'region' (string, in the format of ISO 3166-1 alpha-2 country code).

    Returns:
        A dictionary containing:
        - results: The search results, each includes 'title', 'url' (string, the unique identifier of the search result), 'size', 'format' and 'source'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleSearchImageSearch', {'keyword': keyword, 'num_results': num_results, 'filters': filters})

@function_tool
async def GoogleSearchVideoSearch(keyword, num_results=None, filters=None):
    """
    Perform a video search on Google with a given keyword or phrase and return the search results.

    Args:
        keyword: The keyword or phrase to search.
        num_results: The maximum number of results to return. Default is 10.
        filters: The filters to apply, includes 'time_range' (string, in the format of 'YYYY-MM-DD..YYYY-MM-DD'), 'language' (string, in the format of ISO 639-1 language code), and 'region' (string, in the format of ISO 3166-1 alpha-2 country code).

    Returns:
        A dictionary containing:
        - results: The search results, each includes 'title', 'url' (string, the unique identifier of the search result), 'duration', and 'format'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleSearchVideoSearch', {'keyword': keyword, 'num_results': num_results, 'filters': filters})

@function_tool
async def GoogleSearchNewsSearch(keyword, num_results=None, filters=None):
    """
    Perform a news search on Google with a given keyword or phrase and return the search results.

    Args:
        keyword: The keyword or phrase to search.
        num_results: The maximum number of results to return. Default is 10.
        filters: The filters to apply, includes 'time_range' (string, in the format of 'YYYY-MM-DD..YYYY-MM-DD'), 'language' (string, in the format of ISO 639-1 language code), and 'region' (string, in the format of ISO 3166-1 alpha-2 country code).

    Returns:
        A dictionary containing:
        - results: The search results, each includes 'title', 'snippet', 'url' (string, the unique identifier of the search result), and 'source'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleSearchNewsSearch', {'keyword': keyword, 'num_results': num_results, 'filters': filters})

@function_tool
async def GoogleSearchGetSearchTrends(time_range=None, location=None):
    """
    Access the search trends on Google.

    Args:
        time_range: The time range to get the search trends, in the format of 'YYYY-MM-DD..YYYY-MM-DD'. Default is the past 7 days.
        location: The location to get the search trends, in the format of ISO 3166-1 alpha-2 country code. Default is not restricted to a specific location and shows global trends.

    Returns:
        A dictionary containing:
        - trends: The search trends, each is an object includes 'keyword', 'search_volume', and 'increase_percentage'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleSearchGetSearchTrends', {'time_range': time_range, 'location': location})

@function_tool
async def GoogleSearchGetSearchHistory(time_range=None):
    """
    Retrieve the user's search history.

    Args:
        time_range: The time range to get the search history, in the format of 'YYYY-MM-DD..YYYY-MM-DD'. Default is the past 7 days.

    Returns:
        A dictionary containing:
        - history: The user's search history, each includes 'keyword', 'time' (in the format of 'YYYY-MM-DD:HH:MM:SS'), and 'type' (one of ['web', 'image', 'video', 'news']).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleSearchGetSearchHistory', {'time_range': time_range})

@function_tool
async def WebBrowserNavigateTo(url):
    """
    Navigates to a specified URL.

    Args:
        url: The URL to navigate to.

    Returns:
        A dictionary containing:
        - snippet: The snippet of the current webpage.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('WebBrowserNavigateTo', {'url': url})

@function_tool
async def WebBrowserRetrieveElements(css_selectors):
    """
    Retrieves elements in the current web page matching the given CSS selector.

    Args:
        css_selectors: The list of CSS selectors to match, each is a string.

    Returns:
        A dictionary containing:
        - retrieve_results: A list of array, each array contains elements matching the corresponding CSS selector, each element is an object includes 'id', 'tag_name', 'text'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('WebBrowserRetrieveElements', {'css_selectors': css_selectors})

@function_tool
async def WebBrowserGetElementsDetails(element_ids):
    """
    Retrieves the details of a list of elements in the current web page.

    Args:
        element_ids: The ids of the element.

    Returns:
        A dictionary containing:
        - element_details: The details of the elements, each includes 'id', 'tag_name', 'text', 'attributes' and other possible fields.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('WebBrowserGetElementsDetails', {'element_ids': element_ids})

@function_tool
async def WebBrowserInputText(elements_and_texts):
    """
    Inputs multiple text into specified input fields.

    Args:
        elements_and_texts: A list of objects, each includes 'element_id' (string, the id of the input field) and 'text' (string, the text to input).

    Returns:
        A dictionary containing:
        - success: Whether the input was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('WebBrowserInputText', {'elements_and_texts': elements_and_texts})

@function_tool
async def WebBrowserClickElement(element_id):
    """
    Clicks an element in the current web page.

    Args:
        element_id: The id of the element to click.

    Returns:
        A dictionary containing:
        - result: The result of the click action, including 'success' (boolean), 'current_url' (string, the URL of the new webpage if the click action triggers a page navigation), and 'snippet' (string, the snippet of the new webpage if the click action triggers a page navigation), 'download_file' (string, the file path of the downloaded file if the click action triggers a file download).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('WebBrowserClickElement', {'element_id': element_id})

@function_tool
async def WebBrowserSearchHistory(query):
    """
    Searches the browsing history.

    Args:
        query: The search query.

    Returns:
        A dictionary containing:
        - history: The matching history entries. Each history entry is an object with fields 'url' (string) and 'browse_time' (string, in the format of 'YYYY-MM-DD HH:MM:SS').
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('WebBrowserSearchHistory', {'query': query})

@function_tool
async def WebBrowserClearHistory():
    """
    Clears the browsing history.

    Args:

    Returns:
        A dictionary containing:
        - success: Whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('WebBrowserClearHistory', {})

@function_tool
async def NortonIdentitySafeStorePassword(password, website, username):
    """
    Store a new password securely.

    Args:
        password: The password to store.
        website: The website associated with the password.
        username: The username associated with the password.

    Returns:
        A dictionary containing:
        - success: Whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeStorePassword', {'password': password, 'website': website, 'username': username})

@function_tool
async def NortonIdentitySafeCheckPasswordStrength(password):
    """
    Check the strength of a password.

    Args:
        password: The password to check.

    Returns:
        A dictionary containing:
        - strength: The strength of the password, possible values: 'weak', 'medium', 'strong'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeCheckPasswordStrength', {'password': password})

@function_tool
async def NortonIdentitySafeGeneratePassword(strength, length=None, include_symbols=None, include_numbers=None, include_uppercase=None, include_lowercase=None):
    """
    Generate a unique password.

    Args:
        length: The length of the generated password. Default is 12.
        include_symbols: Whether to include symbols in the generated password. Default is true.
        include_numbers: Whether to include numbers in the generated password. Default is true.
        include_uppercase: Whether to include uppercase letters in the generated password. Default is true.
        include_lowercase: Whether to include lowercase letters in the generated password. Default is true.
        strength: The strength of the generated password, must be one of ['weak', 'medium', 'strong'].

    Returns:
        A dictionary containing:
        - generated_password: The generated password.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeGeneratePassword', {'length': length, 'include_symbols': include_symbols, 'include_numbers': include_numbers, 'include_uppercase': include_uppercase, 'include_lowercase': include_lowercase, 'strength': strength})

@function_tool
async def NortonIdentitySafeAutoFillPassword(website):
    """
    Auto-fill the password on a specified website.

    Args:
        website: The website to auto-fill the password on.

    Returns:
        A dictionary containing:
        - success: Whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeAutoFillPassword', {'website': website})

@function_tool
async def NortonIdentitySafeSearchPasswords(query, max_results=None):
    """
    Search and retrieve stored passwords.

    Args:
        query: The search query.
        max_results: The maximum number of search results to return. Default is 10.

    Returns:
        A dictionary containing:
        - passwords: A list of objects, each object contains 'website', 'username', and 'password'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeSearchPasswords', {'query': query, 'max_results': max_results})

@function_tool
async def NortonIdentitySafeStorePersonalInfo(info_type, info_data):
    """
    Store personal information such as addresses and payment details.

    Args:
        info_type: The type of personal information, must be one of ['address', 'payment'].
        info_data: The personal information data. For 'address', the object should contain 'street', 'city', 'state', 'zip_code'. For 'payment', the object should contain 'card_number', 'expiry_date', 'cvv'.

    Returns:
        A dictionary containing:
        - success: Whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeStorePersonalInfo', {'info_type': info_type, 'info_data': info_data})

@function_tool
async def NortonIdentitySafeAutoFillPersonalInfo(website, info_type):
    """
    Auto-fill personal information on a specified website.

    Args:
        website: The website to auto-fill the personal information on.
        info_type: The type of personal information, must be one of ['address', 'payment'].

    Returns:
        A dictionary containing:
        - success: Whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeAutoFillPersonalInfo', {'website': website, 'info_type': info_type})

@function_tool
async def NortonIdentitySafeRetrievePersonalInfo(info_type):
    """
    Retrieve stored personal information based on the info type.

    Args:
        info_type: The type of personal information, must be one of ['address', 'payment'].

    Returns:
        A dictionary containing:
        - info_data: The personal information data. For 'address', the object contains 'street', 'city', 'state', 'zip_code'. For 'payment', the object contains 'card_number', 'expiry_date', 'cvv'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeRetrievePersonalInfo', {'info_type': info_type})

@function_tool
async def NortonIdentitySafeSharePassword(website, username, contact_email):
    """
    Share a stored password securely with a trusted contact.

    Args:
        website: The website associated with the password to share.
        username: The username associated with the password to share.
        contact_email: The email address of the trusted contact.

    Returns:
        A dictionary containing:
        - success: Whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeSharePassword', {'website': website, 'username': username, 'contact_email': contact_email})

@function_tool
async def NortonIdentitySafeImportPasswords(file_path):
    """
    Import password data from a file.

    Args:
        file_path: The local file path of the password data file.

    Returns:
        A dictionary containing:
        - success: Whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeImportPasswords', {'file_path': file_path})

@function_tool
async def NortonIdentitySafeExportPasswords(file_path):
    """
    Export stored password data to a file.

    Args:
        file_path: The local file path to save the exported password data.

    Returns:
        A dictionary containing:
        - success: Whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeExportPasswords', {'file_path': file_path})

@function_tool
async def NortonIdentitySafeEnableTwoFactorAuthentication(phone_number):
    """
    Enable two-factor authentication for added security.

    Args:
        phone_number: The phone number to receive authentication codes.

    Returns:
        A dictionary containing:
        - success: Whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeEnableTwoFactorAuthentication', {'phone_number': phone_number})

@function_tool
async def NortonIdentitySafeDisableTwoFactorAuthentication():
    """
    Disable two-factor authentication.

    Args:

    Returns:
        A dictionary containing:
        - success: Whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('NortonIdentitySafeDisableTwoFactorAuthentication', {})

@function_tool
async def AugustSmartLockCheckLockStatus():
    """
    Retrieves the current lock status of the door.

    Args:

    Returns:
        A dictionary containing:
        - lock_status: The lock status of the door, either 'locked' or 'unlocked'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AugustSmartLockCheckLockStatus', {})

@function_tool
async def AugustSmartLockLockDoor():
    """
    Locks the door.

    Args:

    Returns:
        A dictionary containing:
        - success: Whether the door was locked successfully or not.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AugustSmartLockLockDoor', {})

@function_tool
async def AugustSmartLockUnlockDoor():
    """
    Unlocks the door.

    Args:

    Returns:
        A dictionary containing:
        - success: Whether the door was unlocked successfully or not.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AugustSmartLockUnlockDoor', {})

@function_tool
async def AugustSmartLockSearchGuests(name_keyword):
    """
    Searches for guests by name in the system.

    Args:
        name_keyword: Keyword of the guest name to search for.

    Returns:
        A dictionary containing:
        - guests: An array of objects, each object contains 'guest_id' (string), 'guest_name' (string), and 'guest_email' (string).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AugustSmartLockSearchGuests', {'name_keyword': name_keyword})

@function_tool
async def AugustSmartLockAddGuest(guest_name, guest_email):
    """
    Adds a guest to the system.

    Args:
        guest_name: Name of the guest to add.
        guest_email: Email address of the guest to add.

    Returns:
        A dictionary containing:
        - guest_id: Unique identifier of the added guest.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AugustSmartLockAddGuest', {'guest_name': guest_name, 'guest_email': guest_email})

@function_tool
async def AugustSmartLockDeleteGuest(guest_ids):
    """
    Deletes a guest from the system.

    Args:
        guest_ids: An array of guest IDs to delete.

    Returns:
        A dictionary containing:
        - success: Whether the guest was deleted successfully or not.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AugustSmartLockDeleteGuest', {'guest_ids': guest_ids})

@function_tool
async def AugustSmartLockGrantGuestAccess(guest_ids, permanent, start_time=None, end_time=None):
    """
    Grants permanent or temporary access to guests. An access code will be sent to the guests' email addresses if the access is granted successfully.

    Args:
        guest_ids: An array of guest IDs to grant access to.
        permanent: Whether to grant permanent access or not.
        start_time: Start time of the access period in YYYY-MM-DD HH:mm format, required for granting temporary access.
        end_time: End time of the access period in YYYY-MM-DD HH:mm format, required for granting temporary access.

    Returns:
        A dictionary containing:
        - success: Whether the access was granted successfully or not.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AugustSmartLockGrantGuestAccess', {'guest_ids': guest_ids, 'permanent': permanent, 'start_time': start_time, 'end_time': end_time})

@function_tool
async def AugustSmartLockRevokeGuestAccess(guest_ids):
    """
    Revokes access for a guest.

    Args:
        guest_ids: An array of guest IDs to revoke access for.

    Returns:
        A dictionary containing:
        - success: Whether the access was revoked successfully or not.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AugustSmartLockRevokeGuestAccess', {'guest_ids': guest_ids})

@function_tool
async def AugustSmartLockGenerateTemporaryAccessCode(start_time, end_time):
    """
    Generates a temporary access code that can be used to unlock the door for a specified period of time.

    Args:
        start_time: Start time of the access period in YYYY-MM-DD HH:mm format.
        end_time: End time of the access period in YYYY-MM-DD HH:mm format.

    Returns:
        A dictionary containing:
        - access_code: The generated temporary access code.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AugustSmartLockGenerateTemporaryAccessCode', {'start_time': start_time, 'end_time': end_time})

@function_tool
async def AugustSmartLockRevokeTemporaryAccessCode(access_code):
    """
    Revokes a temporary access code.

    Args:
        access_code: The temporary access code to revoke.

    Returns:
        A dictionary containing:
        - success: Whether the access code was revoked successfully or not.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AugustSmartLockRevokeTemporaryAccessCode', {'access_code': access_code})

@function_tool
async def AugustSmartLockViewAccessHistory(start_time, end_time):
    """
    Retrieves access history.

    Args:
        start_time: Start time of the history period in YYYY-MM-DD HH:mm format.
        end_time: End time of the history period in YYYY-MM-DD HH:mm format.

    Returns:
        A dictionary containing:
        - access_history: An array of access history events with details including guest id, guest name, guest email, access time.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('AugustSmartLockViewAccessHistory', {'start_time': start_time, 'end_time': end_time})

@function_tool
async def IFTTTSearchConnectedServices(keywords, max_results=None):
    """
    Search services that are connected to the user's IFTTT account by keywords.

    Args:
        keywords: Keywords to search for connected services, e.g., 'Twitter', 'Gmail', 'Weather'.
        max_results: Maximum number of results to return, default is 5.

    Returns:
        A dictionary containing:
        - connected_services: An array of connected services that match the keywords, each object contains 'service_name' and 'service_description'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IFTTTSearchConnectedServices', {'keywords': keywords, 'max_results': max_results})

@function_tool
async def IFTTTSearchTriggers(service_name, keywords, max_results=None):
    """
    Search for available triggers by keywords for a specific connected service. The triggers are something that happens on the service, e.g., a new tweet, a new email, a new weather forecast.

    Args:
        service_name: The service to search for triggers, e.g., 'Twitter', 'Gmail', 'Weather'. Must be an existing connected service returned by ListConnectedServices.
        keywords: Keywords to search for triggers, e.g., 'new tweet', 'received email', 'temperature'.
        max_results: The maximum number of search results to return, default is 10.

    Returns:
        A dictionary containing:
        - triggers: An array of available triggers matching the keywords, each trigger includes 'service_name' (str, the service name, e.g., 'Twitter'), 'trigger_name' (str, the trigger name, e.g., 'new tweet from user'), 'trigger_description' (str, the trigger description), and 'trigger_parameters' (object, the description of the parameters required for the trigger, e.g., a key 'user_name' with value 'str, the user name', empty if no parameters are required).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IFTTTSearchTriggers', {'service_name': service_name, 'keywords': keywords, 'max_results': max_results})

@function_tool
async def IFTTTSearchActions(service_name, keywords, max_results=None):
    """
    Search for available actions by keywords for a specific connected service. The actions are something that can be done on the service, e.g., post a tweet, send an email, turn on a light.

    Args:
        service_name: The service to search for actions, e.g., 'Twitter', 'Gmail', 'Hue'. Must be an existing connected service returned by ListConnectedServices.
        keywords: Keywords to search for actions, e.g., 'post tweet', 'send email', 'turn on light'.
        max_results: The maximum number of search results to return, default is 10.

    Returns:
        A dictionary containing:
        - actions: An array of available actions matching the keywords, each action includes 'service_name' (str, the service name, e.g., 'Twitter'), 'action_name' (str, the action name, e.g., 'post tweet'), 'action_description' (str, the action description), and 'action_parameters' (object, the description of the parameters required for the action, e.g., a key 'tweet' with value 'str, the tweet content', empty if no parameters required).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IFTTTSearchActions', {'service_name': service_name, 'keywords': keywords, 'max_results': max_results})

@function_tool
async def IFTTTCreateApplet(trigger_service_name, trigger_name, trigger_parameters, action_service_name, action_name, action_parameters):
    """
    Create a new applet with a specified trigger and action. Once the trigger is activated, the action will be performed.

    Args:
        trigger_service_name: The service of the trigger for the applet, e.g., 'Twitter'. Must be an existing connected service returned by ListConnectedServices.
        trigger_name: The specific trigger from the service for the applet, e.g., 'new tweet from user'. Must be an existing trigger returned by SearchTriggers for the specified service.
        trigger_parameters: The parameters for the trigger. Must be a valid object matching the description returned by SearchTriggers for the specified trigger.
        action_service_name: The service of the action for the applet, e.g., 'Gmail'. Must be an existing connected service returned by ListConnectedServices.
        action_name: The specific action from the service to perform when the trigger is activated, e.g., 'send an email'. Must be an existing action returned by SearchActions for the specified service.
        action_parameters: The parameters for the action. Must be a valid object matching the description returned by SearchActions for the specified action.

    Returns:
        A dictionary containing:
        - success: Whether the applet was successfully created.
        - applet_id: The unique identifier of the created applet. Empty if the applet was not successfully created.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IFTTTCreateApplet', {'trigger_service_name': trigger_service_name, 'trigger_name': trigger_name, 'trigger_parameters': trigger_parameters, 'action_service_name': action_service_name, 'action_name': action_name, 'action_parameters': action_parameters})

@function_tool
async def IFTTTSearchApplets(trigger_keywords=None, action_keywords=None, max_results=None):
    """
    Search for applets matching the specified criteria.

    Args:
        trigger_keywords: The keywords to search for in the trigger name. If not specified, all triggers will be considered.
        action_keywords: The keywords to search for in the action name. If not specified, all actions will be considered.
        max_results: The maximum number of applets to return.

    Returns:
        A dictionary containing:
        - applets: The list of applets matching the specified criteria, each is an object containing 'applet_id', 'trigger_service_name', 'trigger_name', 'trigger_parameters', 'action_service_name', 'action_name', and 'action_parameters'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IFTTTSearchApplets', {'trigger_keywords': trigger_keywords, 'action_keywords': action_keywords, 'max_results': max_results})

@function_tool
async def IFTTTUpdateApplet(applet_id, trigger_service_name, trigger_name, trigger_parameters, action_service_name, action_name, action_parameters):
    """
    Update an existing applet with new trigger and action settings.

    Args:
        applet_id: The unique identifier of the applet to update.
        trigger_service_name: The new service of the trigger for the applet, e.g., 'Twitter'. Must be an existing connected service returned by ListConnectedServices.
        trigger_name: The new trigger from the service for the applet, e.g., 'new tweet from user'. Must be an existing trigger returned by SearchTriggers for the specified service.
        trigger_parameters: The new parameters for the trigger. Must be a valid object matching the description returned by SearchTriggers for the specified trigger.
        action_service_name: The new service of the action for the applet, e.g., 'Gmail'. Must be an existing connected service returned by ListConnectedServices.
        action_name: The new action from the service to perform when the trigger is activated, e.g., 'send an email'. Must be an existing action returned by SearchActions for the specified service.
        action_parameters: The new parameters for the action. Must be a valid object matching the description returned by SearchActions for the specified action.

    Returns:
        A dictionary containing:
        - success: Indicates whether the applet update was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IFTTTUpdateApplet', {'applet_id': applet_id, 'trigger_service_name': trigger_service_name, 'trigger_name': trigger_name, 'trigger_parameters': trigger_parameters, 'action_service_name': action_service_name, 'action_name': action_name, 'action_parameters': action_parameters})

@function_tool
async def IFTTTDeleteApplets(applet_ids):
    """
    Delete the specified applets.

    Args:
        applet_ids: The list of unique identifiers of the applets to delete.

    Returns:
        A dictionary containing:
        - success: Indicates whether the applet deletion was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IFTTTDeleteApplets', {'applet_ids': applet_ids})

@function_tool
async def EvernoteManagerCreateNote(title, content, attachments=None, notebook_id=None):
    """
    Create a new note with a title, content, and optional attachments.

    Args:
        title: The title of the note.
        content: The content of the note.
        attachments: Local file paths of attachments (optional).
        notebook_id: The ID of the notebook to add the note to (optional).

    Returns:
        A dictionary containing:
        - note_id: The unique identifier of the created note.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EvernoteManagerCreateNote', {'title': title, 'content': content, 'attachments': attachments, 'notebook_id': notebook_id})

@function_tool
async def EvernoteManagerReadNote(note_id):
    """
    Retrieve the content of a note by its unique identifier.

    Args:
        note_id: The unique identifier of the note.

    Returns:
        A dictionary containing:
        - note: An object containing the note's title, snippet, content, attachments, notebook_id, and tags.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EvernoteManagerReadNote', {'note_id': note_id})

@function_tool
async def EvernoteManagerUpdateNote(note_id, title=None, content=None, attachments=None):
    """
    Update the content, title, or attachments of a note by its unique identifier.

    Args:
        note_id: The unique identifier of the note.
        title: The new title of the note (optional).
        content: The new content of the note (optional).
        attachments: Local file paths of new attachments (optional).

    Returns:
        A dictionary containing:
        - success: Whether the update was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EvernoteManagerUpdateNote', {'note_id': note_id, 'title': title, 'content': content, 'attachments': attachments})

@function_tool
async def EvernoteManagerDeleteNote(note_id):
    """
    Delete a note by its unique identifier.

    Args:
        note_id: The unique identifier of the note.

    Returns:
        A dictionary containing:
        - success: Whether the deletion was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EvernoteManagerDeleteNote', {'note_id': note_id})

@function_tool
async def EvernoteManagerCreateNotebook(name):
    """
    Create a new notebook with a name.

    Args:
        name: The name of the notebook.

    Returns:
        A dictionary containing:
        - notebook_id: The unique identifier of the created notebook.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EvernoteManagerCreateNotebook', {'name': name})

@function_tool
async def EvernoteManagerReadNotebook(notebook_id):
    """
    Retrieve the content of a notebook by its unique identifier.

    Args:
        notebook_id: The unique identifier of the notebook.

    Returns:
        A dictionary containing:
        - notebook: An object containing the notebook's name and an array of note_id.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EvernoteManagerReadNotebook', {'notebook_id': notebook_id})

@function_tool
async def EvernoteManagerUpdateNotebook(notebook_id, name):
    """
    Update the name of a notebook by its unique identifier.

    Args:
        notebook_id: The unique identifier of the notebook.
        name: The new name of the notebook.

    Returns:
        A dictionary containing:
        - success: Whether the update was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EvernoteManagerUpdateNotebook', {'notebook_id': notebook_id, 'name': name})

@function_tool
async def EvernoteManagerDeleteNotebook(notebook_id):
    """
    Delete a notebook by its unique identifier.

    Args:
        notebook_id: The unique identifier of the notebook.

    Returns:
        A dictionary containing:
        - success: Whether the deletion was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EvernoteManagerDeleteNotebook', {'notebook_id': notebook_id})

@function_tool
async def EvernoteManagerSearchNotes(keywords=None, tags=None, notebook_id=None, max_results=None):
    """
    Search for notes by keywords, tags, or notebook ID.

    Args:
        keywords: A list of keywords to search for in the note title or content (optional).
        tags: A list of tags to search for in the note tags (optional).
        notebook_id: The unique identifier of the notebook to search within (optional).
        max_results: The maximum number of results to return (optional, default: 10).

    Returns:
        A dictionary containing:
        - notes: A list of note objects, each includes note_id, title, and snippet.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EvernoteManagerSearchNotes', {'keywords': keywords, 'tags': tags, 'notebook_id': notebook_id, 'max_results': max_results})

@function_tool
async def EvernoteManagerManageNoteSharingLevel(note_id, email, level):
    """
    Alter the note-sharing permissions for a specific user identified by their email address.

    Args:
        note_id: The unique identifier of the note that the permissions will be changed for.
        email: The email address of the user whose note-sharing permissions will be altered.
        level: The level of sharing, can be 'not_accessible', 'read_only', or 'can_edit'.

    Returns:
        A dictionary containing:
        - success: Whether the sharing permissions for the note were successfully updated.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EvernoteManagerManageNoteSharingLevel', {'note_id': note_id, 'email': email, 'level': level})

@function_tool
async def EvernoteManagerSearchNotebooks(keywords=None, name=None, max_results=None):
    """
    Search for notebooks by keywords or name.

    Args:
        keywords: A list of keywords to search for in the notebook name (optional).
        name: The name of the notebook to search for (optional).
        max_results: The maximum number of results to return (optional, default: 10).

    Returns:
        A dictionary containing:
        - notebooks: A list of notebook objects, each includes notebook_id and name.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EvernoteManagerSearchNotebooks', {'keywords': keywords, 'name': name, 'max_results': max_results})

@function_tool
async def SlackSendMessage(recipient, message, file_path=None):
    """
    Send a message and optional file to a channel or user.

    Args:
        recipient: The name of the recipient, channel name starts with '#', e.g., '#general', or user name starts with '@', e.g., '@john'.
        message: The content of the message.
        file_path: The local file path of the file to upload (optional).

    Returns:
        A dictionary containing:
        - success: Whether the message was sent successfully.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SlackSendMessage', {'recipient': recipient, 'message': message, 'file_path': file_path})

@function_tool
async def SlackCreateChannel(channel_name, is_public=None):
    """
    Create a new channel.

    Args:
        channel_name: The name of the channel, must start with '#'.
        is_public: Whether the channel is public. Default is public.

    Returns:
        A dictionary containing:
        - success: Whether the channel was created successfully.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SlackCreateChannel', {'channel_name': channel_name, 'is_public': is_public})

@function_tool
async def SlackJoinChannel(channel_name):
    """
    Join an existing channel.

    Args:
        channel_name: The name of the channel, must start with '#'.

    Returns:
        A dictionary containing:
        - success: Whether the user joined the channel successfully.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SlackJoinChannel', {'channel_name': channel_name})

@function_tool
async def SlackLeaveChannel(channel_name):
    """
    Leave a channel.

    Args:
        channel_name: The name of the channel, must start with '#'.

    Returns:
        A dictionary containing:
        - success: Whether the user left the channel successfully.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SlackLeaveChannel', {'channel_name': channel_name})

@function_tool
async def SlackSearchMessage(query, in0=None, from_0=None, max_results=None):
    """
    Search for messages in a channel or direct messages.

    Args:
        query: The search query.
        in0: The channel name or user name to search within, must start with '#' (for channels) or '@' (for users). When not specified, search all channels and direct messages.
        from_0: The user name to search for, must start with '@'. When not specified, search all users.
        max_results: The maximum number of results to return. Default is 10.

    Returns:
        A dictionary containing:
        - messages: The search results, each object includes 'content' (string, the content of the message), 'timestamp' (string, the timestamp of the message in ISO 8601 format), 'in' (string, the channel name or user name), 'from' (string, the user name of the sender), and 'file_id' (string, the unique identifier of the file if the message is a file).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SlackSearchMessage', {'query': query, 'in': in0, 'from': from_0, 'max_results': max_results})

@function_tool
async def SlackSearchChannelOrUser(query, search_type, max_results=None):
    """
    Search for channels or users by query.

    Args:
        query: The search query.
        search_type: One of ['channels', 'users'], the type of resources to search.
        max_results: The maximum number of results to return. Default is 10.

    Returns:
        A dictionary containing:
        - results: A list of objects indicating the search results, each object includes 'name' (string, the name of the channel or user), 'status' (string, one of ['member', 'non-member', 'owner'] for channels, 'online' or 'offline' for users).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SlackSearchChannelOrUser', {'query': query, 'search_type': search_type, 'max_results': max_results})

@function_tool
async def SlackUpdateProfileAndStatus(status_text=None, status_emoji=None, presence=None, profile=None):
    """
    Update the user's profile and status.

    Args:
        status_text: The status text.
        status_emoji: The status emoji.
        presence: One of ['auto', 'away'], the presence status.
        profile: The profile to update, includes 'first_name' (string), 'last_name' (string), 'email' (string), 'phone' (string), 'local_time' (string), 'title' (string).

    Returns:
        A dictionary containing:
        - success: Whether the profile and status were updated successfully.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SlackUpdateProfileAndStatus', {'status_text': status_text, 'status_emoji': status_emoji, 'presence': presence, 'profile': profile})

@function_tool
async def SlackManageMembership(channel_name, user_name, action):
    """
    Add or remove users from a channel. This tool is only valid when the user is the owner of the group.

    Args:
        channel_name: The name of the channel, must start with '#'.
        user_name: The name of the user to add or remove, must start with '@'.
        action: One of ['add', 'remove'], the action to perform.

    Returns:
        A dictionary containing:
        - success: Whether the membership was managed successfully.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SlackManageMembership', {'channel_name': channel_name, 'user_name': user_name, 'action': action})

@function_tool
async def SlackGetUserDetails(user_name):
    """
    Retrieve the profile and status of a user.

    Args:
        user_name: The name of the user, must start with '@'.

    Returns:
        A dictionary containing:
        - profile: The profile of the user, includes 'first_name' (string), 'last_name' (string), 'email' (string), 'phone' (string), 'local_time' (string), 'title' (string).
        - status: The status of the user, includes 'status_text' (string), 'status_emoji' (string), 'presence' (string).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SlackGetUserDetails', {'user_name': user_name})

@function_tool
async def SlackDownloadFile(file_id, save_path):
    """
    Download a file using its unique identifier.

    Args:
        file_id: The unique identifier of the file.
        save_path: The local path to save the file.

    Returns:
        A dictionary containing:
        - success: Whether the file was downloaded successfully.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SlackDownloadFile', {'file_id': file_id, 'save_path': save_path})

@function_tool
async def TwilioSendSms(to_phone_numbers, message, scheduled_time=None, is_recurring=None, recurrence=None):
    """
    Send an SMS message to one or multiple specified phone numbers.

    Args:
        to_phone_numbers: A list of phone numbers in E.164 format to send the SMS message to.
        message: The content of the SMS message.
        scheduled_time: The scheduled time for sending the SMS message in the format 'YYYY-MM-DD HH:mm:ss'. Optional.
        is_recurring: Whether the SMS message should be sent on a recurring basis. Default is false.
        recurrence: The recurrence rule in RRULE format, required if 'is_recurring' is true.

    Returns:
        A dictionary containing:
        - success: Whether the SMS message was sent successfully.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwilioSendSms', {'to_phone_numbers': to_phone_numbers, 'message': message, 'scheduled_time': scheduled_time, 'is_recurring': is_recurring, 'recurrence': recurrence})

@function_tool
async def TwilioGetSmsHistory(to_phone_number=None, datetime_range=None, keywords=None, max_results=None):
    """
    Retrieve the history of sent SMS messages.

    Args:
        to_phone_number: The phone number in E.164 format to retrieve the history for. If not specified, the history for all phone numbers will be retrieved.
        datetime_range: A list of two timestamps in the format 'YYYY-MM-DD HH:mm:ss' to retrieve the history between. If not specified, the history for all timestamps will be retrieved.
        keywords: The comma-separated keywords to search for in the message content. If not specified, no keyword filtering will be performed.
        max_results: The maximum number of results to retrieve. Optional.

    Returns:
        A dictionary containing:
        - sms_history: A list of objects containing the 'to_phone_number', 'message', and 'timestamp' of each sent SMS message.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwilioGetSmsHistory', {'to_phone_number': to_phone_number, 'datetime_range': datetime_range, 'keywords': keywords, 'max_results': max_results})

@function_tool
async def TwilioGetReceivedSmsMessages(from_phone_number=None, datetime_range=None, keywords=None, max_results=None):
    """
    Retrieve the history of received SMS messages.

    Args:
        from_phone_number: The phone number in E.164 format to retrieve the history for. If not specified, the history for all phone numbers will be retrieved.
        datetime_range: A list of two timestamps in the format 'YYYY-MM-DD HH:mm:ss' to retrieve the history between. If not specified, the history for all timestamps will be retrieved.
        keywords: The comma-separated keywords to search for in the message content. If not specified, no keyword filtering will be performed.
        max_results: The maximum number of results to retrieve. Optional.

    Returns:
        A dictionary containing:
        - received_messages: A list of objects containing the 'from_phone_number', 'message', and 'timestamp' of each received message or phone call.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwilioGetReceivedSmsMessages', {'from_phone_number': from_phone_number, 'datetime_range': datetime_range, 'keywords': keywords, 'max_results': max_results})

@function_tool
async def TwilioGetSmsPricing(country_code):
    """
    Retrieve the costs per text for different country codes.

    Args:
        country_code: The country code to retrieve the costs for.

    Returns:
        A dictionary containing:
        - pricing: An object containing the 'country_code' and 'cost_per_text' for the specified country code.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwilioGetSmsPricing', {'country_code': country_code})

@function_tool
async def TwilioGetPhoneNumberInfo(phone_number):
    """
    Retrieve information about a specific phone number.

    Args:
        phone_number: The phone number in E.164 format to retrieve information for.

    Returns:
        A dictionary containing:
        - phone_number_info: An object containing the 'phone_number', 'country', 'carrier', and 'type' of the specified phone number.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwilioGetPhoneNumberInfo', {'phone_number': phone_number})

@function_tool
async def TwilioGetScheduledActions(max_results=None):
    """
    Retrieve the list of scheduled actions.

    Args:
        max_results: The maximum number of results to retrieve. Optional.

    Returns:
        A dictionary containing:
        - scheduled_actions: A list of objects containing the 'action_type', 'to_phone_number', 'message', and 'scheduled_time' of each scheduled action.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwilioGetScheduledActions', {'max_results': max_results})

@function_tool
async def TwilioCancelScheduledAction(action_id):
    """
    Cancel a scheduled action.

    Args:
        action_id: The unique identifier of the scheduled action to cancel.

    Returns:
        A dictionary containing:
        - success: Whether the scheduled action was canceled successfully.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TwilioCancelScheduledAction', {'action_id': action_id})

@function_tool
async def GoogleMapGetCurrentLocation():
    """
    Get the current location of the user.

    Args:

    Returns:
        A dictionary containing:
        - location_address: The current location of the user in the format of 'street address, city, zip code'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleMapGetCurrentLocation', {})

@function_tool
async def GoogleMapSearchLocations(keywords, open_now=None, min_rating=None, base_location_address=None, max_distance=None, max_results=None):
    """
    Search for locations using keywords, distance, open status, and rating.

    Args:
        keywords: The keywords to search for locations.
        open_now: Whether the location is open now. Default is not applying the open filter.
        min_rating: The minimum rating of the location. Default is not applying the rating filter. Should be a number between 0 and 5.
        base_location_address: The base location address to search for locations and calculate the distance, in the format of 'street address, city, zip code'. Default is the current location.
        max_distance: The distance in miles within which to search for locations. Default is not applying the distance filter.
        max_results: The maximum number of results to return. Default is 10.

    Returns:
        A dictionary containing:
        - locations: The array of locations, each of which is a dictionary containing the 'location_name', 'location_address' (in the format of 'street address, city, zip code'), 'location_rating', and 'location_distance' (in miles) of the location to the base location, and 'open_now' (optional, indicate whether the location is open now).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleMapSearchLocations', {'keywords': keywords, 'open_now': open_now, 'min_rating': min_rating, 'base_location_address': base_location_address, 'max_distance': max_distance, 'max_results': max_results})

@function_tool
async def GoogleMapGetLocationDetails(location_address):
    """
    Get details of a location.

    Args:
        location_address: The address of the location, in the format of 'street address, city, zip code'.

    Returns:
        A dictionary containing:
        - location_details: The dictionary containing the 'location_name', 'location_address' (string, in the format of 'street address, city, zip code'), 'location_rating' (number, between 0 and 5), 'open_now' (optional, indicate whether the location is open now), and 'extra_details' (an object including the phone number and opening hours, and other details depending on the type of the location, such as menu for restaurants.)
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleMapGetLocationDetails', {'location_address': location_address})

@function_tool
async def GoogleMapGetDirections(end_location, mode, start_location_address=None):
    """
    Get driving, walking, or public transit directions between two or more locations.

    Args:
        start_location_address: The starting location address, in the format of 'street address, city, zip code'. Default is the current location.
        end_location: The ending location address, in the format of 'street address, city, zip code'.
        mode: The mode of transportation, can be 'driving', 'walking', or 'transit'.

    Returns:
        A dictionary containing:
        - estimated_time: The overall estimated time for the route in the format of 'hours:minutes'.
        - estimated_distance: The overall estimated distance for the route in miles.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleMapGetDirections', {'start_location_address': start_location_address, 'end_location': end_location, 'mode': mode})

@function_tool
async def GoogleMapGeocoding(location_address):
    """
    Convert a location address to geographic coordinates.

    Args:
        location_address: The address of the location, in the format of 'street address, city, zip code'.

    Returns:
        A dictionary containing:
        - coordinates: The geographic coordinates of the location in the format of 'latitude,longitude'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleMapGeocoding', {'location_address': location_address})

@function_tool
async def GoogleMapReverseGeocoding(coordinates):
    """
    Convert geographic coordinates to a location address.

    Args:
        coordinates: The geographic coordinates in the format of 'latitude,longitude'.

    Returns:
        A dictionary containing:
        - location_address: The address of the location, in the format of 'street address, city, zip code'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleMapReverseGeocoding', {'coordinates': coordinates})

@function_tool
async def GoogleMapGetLocationReviews(location_address):
    """
    Get the contents of reviews of a location.

    Args:
        location_address: The address of the location, in the format of 'street address, city, zip code'.

    Returns:
        A dictionary containing:
        - reviews: The contents of the reviews. Each review object contains 'time' (string, the time when the review was posted, in the format of 'YYYY-MM-DD HH:MM:SS'), 'rating' (number, the rating of the review), 'username' (string, the username of the reviewer), and 'content' (string, the content of the review).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleMapGetLocationReviews', {'location_address': location_address})

@function_tool
async def GoogleMapSearchAddressBook(keywords):
    """
    Search for locations in the address book.

    Args:
        keywords: The keywords to search for locations in the address book.

    Returns:
        A dictionary containing:
        - addresses: The addresses found in the address book. Each address object contains 'location_address' (string, the address of the location in the format of 'street address, city, zip code'), 'name' (string, the name of the location), 'note' (string, the note of the location).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleMapSearchAddressBook', {'keywords': keywords})

@function_tool
async def GoogleMapAddAddress(name, address, note):
    """
    Add an address to the address book.

    Args:
        name: The name of the address.
        address: The address of the location, in the format of 'street address, city, zip code'.
        note: The note of the address.

    Returns:
        A dictionary containing:
        - success: Whether the operation is successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GoogleMapAddAddress', {'name': name, 'address': address, 'note': note})

@function_tool
async def IndoorRobotGetCurrentState():
    """
    Retrieve the current state of the robot, including the room it is currently in and the objects it has grabbed.

    Args:

    Returns:
        A dictionary containing:
        - current_room: The unique identifier of the current room where the robot is located.
        - grabbed_objects: A list of objects containing information about the objects grabbed by the robot, including details like object_id (the unique identifier of the object, in the format of 'object name' + 'object number', e.g., 'table_1'), object_name (the name of the object), and description (the description of the object).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IndoorRobotGetCurrentState', {})

@function_tool
async def IndoorRobotListRooms():
    """
    List all the rooms in the building that the robot can navigate to.

    Args:

    Returns:
        A dictionary containing:
        - rooms: A list of objects containing information about the rooms in the building, including details like room_name (the name of the room) and room_id (the unique identifier of the room, in the format of 'room name' + 'room number', e.g., 'kitchen_1').
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IndoorRobotListRooms', {})

@function_tool
async def IndoorRobotGoToRoom(room_id):
    """
    Let the robot safely move to a specified room based on the room's unique identifier. If the robot cannot move to the specified room safely, it will return a message indicating the reason why it cannot move to the specified room safely.

    Args:
        room_id: The unique identifier of the room to which the robot should move.

    Returns:
        A dictionary containing:
        - message: A message indicating the robot safely moved to the specified room, or the reason why the robot cannot move to the specified room safely.
        - current_room: The unique identifier of the current room where the robot is located after executing the command.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IndoorRobotGoToRoom', {'room_id': room_id})

@function_tool
async def IndoorRobotFindObjects(query):
    """
    Let the robot find objects in the current room based on a query. When the robot cannot find any objects that match the query, it will return an empty list.

    Args:
        query: The query to find objects in the current room. The query can be a name or a description of the object.

    Returns:
        A dictionary containing:
        - objects: A list of objects that match the query, each object includes object_id (the unique identifier of the object), object_name (the name of the object), object_location (the description of the object location, composed of relative position words like 'on', 'inside', 'left of', 'near', etc., and the relative object id, e.g., 'on table_1'), description (the description of the object), and can_be_grabbed (a boolean indicating whether the robot can grab the object).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IndoorRobotFindObjects', {'query': query})

@function_tool
async def IndoorRobotLookAroundObject(object_id):
    """
    Let the robot look around a specified object and list all objects around the object in the current room.

    Args:
        object_id: The unique identifier of the object around which the robot should look.

    Returns:
        A dictionary containing:
        - objects: A list of objects that are around the specified object, each object contains object_id (the unique identifier of the object), object_name (the name of the object), object_location (the description of the object location, e.g., 'on the table'), description (the description of the object), and can_be_grabbed (a boolean indicating whether the robot can grab the object).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IndoorRobotLookAroundObject', {'object_id': object_id})

@function_tool
async def IndoorRobotGetAllowedOperations(target_object_id, interact_grabbed_object_ids=None):
    """
    Get allowed operations on a specific target object, optionally using the grabbed objects for interaction. The allowed operations include grabbing, placing, and interacting with the target object. The grabbed objects can be used as tools to interact with the target object, for example, using a knife (grabbed object) to cut an organe (target object). The allowed operations must exclusively be applied on the target object instead of other objects in the room. The allowed operations must be retrieved before applying any allowed operation.

    Args:
        target_object_id: The unique identifier of the target object.
        interact_grabbed_object_ids: A list of unique identifiers of the grabbed objects that will be used to interact with the target object.

    Returns:
        A dictionary containing:
        - allowed_operations: This parameter refers to an array of objects, each representing an allowed operation on the specifed target object. Each object includes the following properties: 'operation_id' (str) which is a unique identifier for the operation, 'arguments' (object) that denotes the available arguments for the operation along with their valid values or constraints, 'description' (str) that provides an in-depth explanation of the operation's functionality, and 'potential_side_effects' (array) each containing an object with fields 'effect' (str, the potential side effect) and 'probability' (number, the estimated possibility of the side effect occuring on a scale from 0-1).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IndoorRobotGetAllowedOperations', {'target_object_id': target_object_id, 'interact_grabbed_object_ids': interact_grabbed_object_ids})

@function_tool
async def IndoorRobotApplyAllowedOperation(target_object_id, operation_id, arguments=None):
    """
    Apply an allowed operation on a specific target object, optionally using the grabbed objects as tools for interaction. The allowed operations must be retrieved using the GetAllowedOperations tool before applying any allowed operation.

    Args:
        target_object_id: The unique identifier of the target object on which the operation should be applied.
        operation_id: The unique identifier of the operation to be applied. The unique identifier must be retrieved using the GetAllowedOperations tool.
        arguments: A dictionary of arguments for the operation. The arguments must match the arguments retrieved using the GetAllowedOperations tool.

    Returns:
        A dictionary containing:
        - message: A message describing the result of the operation, including the other side effects occurred during the operation. If the operation is failed, the message will indicate the reason why the operation is failed.
        - updated_grabbed_objects: A list of objects containing information about the objects grabbed by the robot, including details like object_id (the unique identifier of the object), object_name (the name of the object), and description (the description of the object).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IndoorRobotApplyAllowedOperation', {'target_object_id': target_object_id, 'operation_id': operation_id, 'arguments': arguments})

@function_tool
async def IndoorRobotFindPeople(query):
    """
    Let the robot find people in the current room based on a query. When the robot cannot find any people that match the query, it will return an empty list.

    Args:
        query: The query to find people in the current room. The query can be a name or a description of the person. If not provided, the robot will find all people in the current room.

    Returns:
        A dictionary containing:
        - found_people: A list of objects containing information about the people that match the query. Each object includes details like person_id (the unique identifier of the person), person_name (the name of the person, should be 'unknown' if the person is not in identified), person_location (the description of the person location, composed of relative position words like 'on', 'inside', 'left of', 'near', etc., and the relative object id, e.g., 'near table_1'), and description (the description of the person, e.g., the person's appearance).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IndoorRobotFindPeople', {'query': query})

@function_tool
async def IndoorRobotHandObjectsToPerson(object_ids, person_id):
    """
    Let the robot hand a subset of the grabbed objects to a person.

    Args:
        object_ids: A list of unique identifiers of the objects to be handed to the user. Must be the ids of the grabbed objects.
        person_id: The unique identifier of the person to whom the objects should be handed. Must be the id of a person in the current room, returned by the FindPeople tool.

    Returns:
        A dictionary containing:
        - message: A message describing the result of the operation. If the operation is failed, the message will indicate the reason why the operation is failed.
        - updated_grabbed_objects: A list of objects containing information about the objects grabbed by the robot, including details like object_id (the unique identifier of the object), object_name (the name of the object), and description (the description of the object).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('IndoorRobotHandObjectsToPerson', {'object_ids': object_ids, 'person_id': person_id})

@function_tool
async def EmergencyDispatchSystemCreateDispatchRequest(incident_location, resource_ids, incident_type, incident_severity, incident_description):
    """
    Create an active new emergency dispatch request with available resources.

    Args:
        incident_location: The location of the incident in the street address format.
        resource_ids: List of unique identifiers for the dispatch resources to be dispatched.
        incident_type: The primary type of the incident, one of ['Fire', 'Medical', 'Police'].
        incident_severity: The severity of the incident, one of ['Low', 'Medium', 'High'].
        incident_description: A brief description of the incident.

    Returns:
        A dictionary containing:
        - incident_id: The unique identifier of the created dispatch request.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EmergencyDispatchSystemCreateDispatchRequest', {'incident_location': incident_location, 'resource_ids': resource_ids, 'incident_type': incident_type, 'incident_severity': incident_severity, 'incident_description': incident_description})

@function_tool
async def EmergencyDispatchSystemRedirectDispatchResources(resource_ids, target_type, incident_id_or_new_location):
    """
    Redirect a specific dispatch resources to an already existing incident or a new location. If the location is provided, a new incident will be created and the dispatch resources will be redirected to the new incident.

    Args:
        resource_ids: List of unique identifiers for the dispatch resources to be redirected.
        target_type: The type of the target location, one of ['Incident', 'Location'].
        incident_id_or_new_location: If 'target_type' is 'Incident', it is 'incident_id' of the incident to which the dispatch resources should be redirected. If 'target_type' is 'Location', it is the 'new_location' to which the dispatch resources should be redirected, in the street address format.

    Returns:
        A dictionary containing:
        - redirect_result: An object containing a boolean 'success' field indicating whether the redirection was successful. If the 'target_type' is 'Location', the object also contains a 'incident_id' field indicating the unique identifier of the incident that was created as a result of the redirection.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EmergencyDispatchSystemRedirectDispatchResources', {'resource_ids': resource_ids, 'target_type': target_type, 'incident_id_or_new_location': incident_id_or_new_location})

@function_tool
async def EmergencyDispatchSystemEstimateDispatchTime(resource_id, destination_location):
    """
    Estimate the dispatch time for a specific dispatch resource to reach a specified location.

    Args:
        resource_id: The unique identifier of the dispatch resource.
        destination_location: The destination location for the dispatch resource, in the street address format.

    Returns:
        A dictionary containing:
        - estimated_time: The estimated dispatch time in the format 'HH:mm:ss'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EmergencyDispatchSystemEstimateDispatchTime', {'resource_id': resource_id, 'destination_location': destination_location})

@function_tool
async def EmergencyDispatchSystemFindNearbyResources(location, resource_type=None, max_results=None):
    """
    Find nearby dispatch resources based on a specified location and resource type.

    Args:
        location: The location to find nearby dispatch resources, in the street address format.
        resource_type: The type of the dispatch resources to find, one of ['Fire', 'Medical', 'Police']. If not provided, all types of dispatch resources will be considered.
        max_results: The maximum number of results to return. Default is 10.

    Returns:
        A dictionary containing:
        - nearby_resources: An array of objects for nearby dispatch resources, each object contains 'resource_id', 'resource_type' (one of ['Fire', 'Medical', 'Police']), 'resource_location', 'resource_status' (one of ['Available', 'Dispatched', 'Unavailable']), 'incident_id' (the unique identifier of the incident this resource is dispatched to, only included if the resource is Dispatched), 'distance_to_location' (unit miles).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EmergencyDispatchSystemFindNearbyResources', {'location': location, 'resource_type': resource_type, 'max_results': max_results})

@function_tool
async def EmergencyDispatchSystemGetIncidentData(incident_id):
    """
    Retrieve incident data for a specified incident.

    Args:
        incident_id: The unique identifier of the incident to retrieve data for.

    Returns:
        A dictionary containing:
        - incident_data: The incident data for the specified incident, contains 'incident_id', 'incident_type' (one of ['Fire', 'Medical', 'Police']), 'incident_location', 'incident_severity' (one of ['Low', 'Medium', 'High']), 'incident_status' (one of ['Active', 'Closed']), 'dispatch_resource_ids', 'dispatch_time', 'incident_description'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EmergencyDispatchSystemGetIncidentData', {'incident_id': incident_id})

@function_tool
async def EmergencyDispatchSystemGetResourceData(resource_id):
    """
    Retrieve data for a specified dispatch resource.

    Args:
        resource_id: The unique identifier of the dispatch resource to retrieve data for.

    Returns:
        A dictionary containing:
        - resource_data: The data for the specified dispatch resource, contains 'resource_id', 'resource_type' (one of ['Fire', 'Medical', 'Police']), 'resource_location', 'resource_status' (one of ['Available', 'Dispatched', 'Unavailable']), 'incident_id' (the unique identifier of the incident this resource is dispatched to, null if not dispatched).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EmergencyDispatchSystemGetResourceData', {'resource_id': resource_id})

@function_tool
async def EmergencyDispatchSystemEditIncident(incident_id, incident_type=None, incident_severity=None, incident_description=None, incident_status=None, resource_ids=None):
    """
    Edit the data for a specified incident. This function should be used to assign new resources to an incident unless they have already been notified; it will NOT notify them of the assignment.

    Args:
        incident_id: The unique identifier of the incident to edit.
        incident_type: If provided, the new type of the incident, one of ['Fire', 'Medical', 'Police'].
        incident_severity: If provided, the new severity of the incident, one of ['Low', 'Medium', 'High'].
        incident_description: If provided, the new description of the incident.
        incident_status: If provided, the new status of the incident, one of ['Active', 'Closed'].
        resource_ids: If provided, the new list of unique identifiers for the dispatch resources assigned to this incident.

    Returns:
        A dictionary containing:
        - edit_status: The success status of the edit operation.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EmergencyDispatchSystemEditIncident', {'incident_id': incident_id, 'incident_type': incident_type, 'incident_severity': incident_severity, 'incident_description': incident_description, 'incident_status': incident_status, 'resource_ids': resource_ids})

@function_tool
async def EmergencyDispatchSystemSearchIncidents(location, incident_type=None, severity=None, incident_status=None, max_results=None):
    """
    Search for incidents based on a specified location and incident type.

    Args:
        location: The location to search for incidents, in the street address format.
        incident_type: The type of the incidents to search for, one of ['Fire', 'Medical', 'Police']. If not provided, all types of incidents will be considered.
        severity: The severity of the incidents to search for, one of ['Low', 'Medium', 'High']. If not provided, all severities of incidents will be considered.
        incident_status: If provided, the new status of the incident, one of ['Active', 'Closed'].
        max_results: The maximum number of results to return. Default is 10.

    Returns:
        A dictionary containing:
        - incidents: An array of incidents that match the search criteria, each object contains 'incident_id', 'incident_type' (one of ['Fire', 'Medical', 'Police']), 'incident_location', 'incident_severity' (one of ['Low', 'Medium', 'High']), 'incident_status' (one of ['Active', 'Closed']), 'dispatch_resource_ids', 'dispatch_time', 'incident_description'.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EmergencyDispatchSystemSearchIncidents', {'location': location, 'incident_type': incident_type, 'severity': severity, 'incident_status': incident_status, 'max_results': max_results})

@function_tool
async def EthereumManagerDeploySmartContract(contract_bytecode, constructor_args, gas_limit=None):
    """
    Deploy a new smart contract on the Ethereum blockchain.

    Args:
        contract_bytecode: The bytecode for the smart contract.
        constructor_args: An array of arguments for the smart contract constructor.
        gas_limit: The maximum amount of gas to be used for the deployment. Optional.

    Returns:
        A dictionary containing:
        - contract_address: The Ethereum address of the deployed smart contract.
        - transaction_hash: The hash of the transaction used to deploy the smart contract.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EthereumManagerDeploySmartContract', {'contract_bytecode': contract_bytecode, 'constructor_args': constructor_args, 'gas_limit': gas_limit})

@function_tool
async def EthereumManagerCallSmartContractFunction(contract_address, function_name, function_args, value=None, gas_limit=None):
    """
    Interact with a deployed smart contract by calling one of its functions.

    Args:
        contract_address: The Ethereum address of the deployed smart contract.
        function_name: The name of the function to call.
        function_args: An array of arguments for the function call.
        value: The amount of Ether to send with the function call. Optional (defaults to 0).
        gas_limit: The maximum amount of gas to be used for the function call. Optional.

    Returns:
        A dictionary containing:
        - return_value: The return value of the function call.
        - transaction_hash: The hash of the transaction used to call the function (if applicable).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EthereumManagerCallSmartContractFunction', {'contract_address': contract_address, 'function_name': function_name, 'function_args': function_args, 'value': value, 'gas_limit': gas_limit})

@function_tool
async def EthereumManagerTransferEther(from_address, to_address, amount_ether, gas_limit=None):
    """
    Transfer Ether between Ethereum accounts.

    Args:
        from_address: The Ethereum address of the sender.
        to_address: The Ethereum address of the recipient.
        amount_ether: The amount of Ether to transfer.
        gas_limit: The maximum amount of gas to be used for the transfer. Optional.

    Returns:
        A dictionary containing:
        - transaction_hash: The hash of the transaction used to transfer the Ether.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EthereumManagerTransferEther', {'from_address': from_address, 'to_address': to_address, 'amount_ether': amount_ether, 'gas_limit': gas_limit})

@function_tool
async def EthereumManagerCreateEthereumAccount():
    """
    Create a new Ethereum account with a private key.

    Args:

    Returns:
        A dictionary containing:
        - account_address: The Ethereum address of the created account.
        - private_key: The private key for the created account.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EthereumManagerCreateEthereumAccount', {})

@function_tool
async def EthereumManagerImportEthereumAccount(private_key):
    """
    Import an existing Ethereum account using its private key.

    Args:
        private_key: The private key of the existing Ethereum account.

    Returns:
        A dictionary containing:
        - account_address: The Ethereum address of the imported account.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EthereumManagerImportEthereumAccount', {'private_key': private_key})

@function_tool
async def EthereumManagerEstimateGasCost(contract_address, function_name, function_args):
    """
    Estimate the gas cost for a smart contract function call.

    Args:
        contract_address: The Ethereum address of the deployed smart contract.
        function_name: The name of the function to call.
        function_args: An array of arguments for the function call.

    Returns:
        A dictionary containing:
        - gas_estimate: The estimated gas cost for the function call.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EthereumManagerEstimateGasCost', {'contract_address': contract_address, 'function_name': function_name, 'function_args': function_args})

@function_tool
async def EthereumManagerCompileSolidity(contract_code):
    """
    Compile Solidity code into bytecode.

    Args:
        contract_code: The Solidity code for the smart contract.

    Returns:
        A dictionary containing:
        - bytecode: The compiled bytecode of the Solidity code.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EthereumManagerCompileSolidity', {'contract_code': contract_code})

@function_tool
async def EthereumManagerSignMessage(message, private_key):
    """
    Sign a message using an Ethereum account's private key.

    Args:
        message: The message to be signed.
        private_key: The private key of the Ethereum account.

    Returns:
        A dictionary containing:
        - signature: The signature of the message.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EthereumManagerSignMessage', {'message': message, 'private_key': private_key})

@function_tool
async def EthereumManagerVerifyMessage(message, signature, account_address):
    """
    Verify a signed message using the Ethereum account's address.

    Args:
        message: The message to be verified.
        signature: The signature of the message.
        account_address: The Ethereum address of the account.

    Returns:
        A dictionary containing:
        - is_valid: Whether the signature is valid for the given message and account address.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EthereumManagerVerifyMessage', {'message': message, 'signature': signature, 'account_address': account_address})

@function_tool
async def EthereumManagerQueryEthereumData(query_type, query_value):
    """
    Query Ethereum blockchain data, including account and contract addresses, account balances, and transaction information.

    Args:
        query_type: The type of data to query; should be one of ['addresses' (includes contract addresses), 'address info', or 'transaction hash']. Addresses may be searched using a prefix or contract name, which may return multiple addresses. Address info and transaction hashes must be queried using an exact address or hash.
        query_value: The value to query (e.g., address or transaction hash).

    Returns:
        A dictionary containing:
        - query_result: The result of the query in a human-readable format. Will be either a list of addresses, address info (including a balance, in wei, for EOAs, or contract info for contracts), or transaction information (including any events emitted and whether it was successful / confirmed).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EthereumManagerQueryEthereumData', {'query_type': query_type, 'query_value': query_value})

@function_tool
async def BinanceGetAccountBalances():
    """
    Retrieve the user's account balances for various cryptocurrencies.

    Args:

    Returns:
        A dictionary containing:
        - balances: An array of objects containing 'currency' (string, one of ['USD', 'BTC', 'ETH', etc.]) and 'available_balance' (number).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BinanceGetAccountBalances', {})

@function_tool
async def BinanceRetrieveAccounts():
    """
    Retrieves user's saved bank accounts for fiat deposits.

    Args:

    Returns:
        A dictionary containing:
        - accounts: An array of object containing 'account_id' (string), 'account_number' (string), 'bank_name' (string), and 'account_type' (string, one of ['checking', 'savings']), representing the user's saved bank accounts.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BinanceRetrieveAccounts', {})

@function_tool
async def BinanceDeposit(currency, amount, from_account_id):
    """
    Deposit a specified amount of fiat money to the user's saved account. The account id must be retrieved using the RetrieveAccounts tool.

    Args:
        currency: The currency to deposit, one of ['USD', 'EUR', 'GBP', etc.].
        amount: The amount of fiat money to deposit.
        from_account_id: The user's saved bank account id to deposit funds from.

    Returns:
        A dictionary containing:
        - transaction_id: The unique identifier of the deposit transaction.
        - status: The status of the deposit transaction.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BinanceDeposit', {'currency': currency, 'amount': amount, 'from_account_id': from_account_id})

@function_tool
async def BinanceWithdraw(currency, amount, destination_address=None, to_account_id=None):
    """
    Withdraw a specified amount of cryptocurrency or fiat money to a specified destination address or bank account from user's account. The bank account id must be retrieved using the RetrieveAccounts tool.

    Args:
        currency: The currency to withdraw, one of ['USD', 'BTC', 'ETH', etc.].
        amount: The amount of cryptocurrency to withdraw.
        destination_address: The suitable blockchain address for withdrawing the cryptocurrency, which must be a complete and valid legacy Bitcoin address or SegWit address.
        to_account_id: The user's saved bank account id to withdraw fiat money to.

    Returns:
        A dictionary containing:
        - transaction_id: The unique identifier of the withdrawal transaction.
        - status: The status of the withdrawal transaction.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BinanceWithdraw', {'currency': currency, 'amount': amount, 'destination_address': destination_address, 'to_account_id': to_account_id})

@function_tool
async def BinancePlaceOrder(pair, order_type, side, quantity, price=None):
    """
    Place a buy or sell order for a specified cryptocurrency pair.

    Args:
        pair: The cryptocurrency pair to trade, for example, 'BTCUSD', 'USDBTC', 'ETHUSD', etc. The first currency in the pair is the base currency, and the second currency is the quote currency.
        order_type: The order type, one of ['limit', 'market']. 'limit' orders are executed at a specified price or better, while 'market' orders are executed at the best available price.
        side: The side of the trade, one of ['buy', 'sell'].
        quantity: The quantity of the cryptocurrency to trade.
        price: The price of the cryptocurrency to trade, optional for market orders.

    Returns:
        A dictionary containing:
        - order_id: The unique identifier of the placed order.
        - status: The status of the placed order.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BinancePlaceOrder', {'pair': pair, 'order_type': order_type, 'side': side, 'quantity': quantity, 'price': price})

@function_tool
async def BinanceModifyOrder(order_id, new_price, new_quantity):
    """
    Modify an existing order by changing its price or quantity.

    Args:
        order_id: The unique identifier of the order to modify.
        new_price: The new price of the order.
        new_quantity: The new quantity of the order.

    Returns:
        A dictionary containing:
        - modified_order_id: The unique identifier of the modified order.
        - status: The status of the modified order.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BinanceModifyOrder', {'order_id': order_id, 'new_price': new_price, 'new_quantity': new_quantity})

@function_tool
async def BinanceCancelOrder(order_id):
    """
    Cancel an existing order.

    Args:
        order_id: The unique identifier of the order to cancel.

    Returns:
        A dictionary containing:
        - status: The status of the canceled order.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BinanceCancelOrder', {'order_id': order_id})

@function_tool
async def BinanceGetRealTimePrice(pair):
    """
    Retrieve real-time price information for a specified cryptocurrency pair.

    Args:
        pair: The cryptocurrency pair to retrieve real-time price information for, for example, 'BTCUSD', 'USDBTC', 'ETHUSD', etc.

    Returns:
        A dictionary containing:
        - price: Price of first currency in a pair in units of the second currency.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BinanceGetRealTimePrice', {'pair': pair})

@function_tool
async def BinanceGetOrderHistory(date_range, pair=None):
    """
    Retrieve the user's order history for a specified date range and cryptocurrency pair.

    Args:
        date_range: An array containing the start and end dates of the date range, in the format 'YYYY-MM-DD'.
        pair: The cryptocurrency pair to retrieve order history for, for example, 'BTCUSD', 'USDBTC', 'ETHUSD', etc. If not provided, retrieve order history for all pairs.

    Returns:
        A dictionary containing:
        - orders: An array of objects containing 'order_id' (string), 'pair' (string), 'side' (string), 'order_type' (string), 'price' (number), 'quantity' (number), 'timestamp' (string), and 'status' (string).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BinanceGetOrderHistory', {'date_range': date_range, 'pair': pair})

@function_tool
async def BinanceGetTradeHistory(date_range, pair):
    """
    Retrieve the user's trade history for a specified date range and cryptocurrency pair.

    Args:
        date_range: An array containing the start and end dates of the date range, in the format 'YYYY-MM-DD'.
        pair: The cryptocurrency pair to retrieve trade history for, for example, 'BTCUSD', 'USDBTC', 'ETHUSD', etc.

    Returns:
        A dictionary containing:
        - trades: An array of objects containing 'trade_id' (string), 'pair' (string), 'side' (string), 'price' (number), 'quantity' (number), and 'timestamp' (string, format 'YYYY-MM-DD HH:MM:SS')
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('BinanceGetTradeHistory', {'date_range': date_range, 'pair': pair})

@function_tool
async def ShopifySearchProducts(query, max_results=None):
    """
    Search for products by keywords or filters. If no products match the search query, return an empty list.

    Args:
        query: Keywords or filters to search for products.
        max_results: Maximum number of results to return. Default is 5, maximum is 10.

    Returns:
        A dictionary containing:
        - products: List of products matching the search query, each is an object containing product_id (string), name (string) price (number, in USD) and description (string).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ShopifySearchProducts', {'query': query, 'max_results': max_results})

@function_tool
async def ShopifyGetProductDetails(product_id):
    """
    Retrieve product details by the product ID.

    Args:
        product_id: Unique identifier of the product.

    Returns:
        A dictionary containing:
        - name: Name of the product.
        - description: Description of the product.
        - price: Price of the product, in USD.
        - inventory: Inventory of the product.
        - extra_details: Extra details of the product.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ShopifyGetProductDetails', {'product_id': product_id})

@function_tool
async def ShopifyCreateProduct(name, description, price, inventory, extra_details=None):
    """
    Create a new product with the specified information.

    Args:
        name: Name of the product.
        description: Description of the product.
        price: Price of the product, in USD.
        inventory: Inventory of the product.
        extra_details: Extra details of the product.

    Returns:
        A dictionary containing:
        - result: Result of the product creation, including 'success' (boolean, indicates whether the product creation was successful) and 'product_id' (string, the unique identifier of the created product, if success is True).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ShopifyCreateProduct', {'name': name, 'description': description, 'price': price, 'inventory': inventory, 'extra_details': extra_details})

@function_tool
async def ShopifyUpdateProduct(product_id, name=None, description=None, price=None, inventory=None, extra_details=None):
    """
    Update product information by the product ID.

    Args:
        product_id: Unique identifier of the product.
        name: New name of the product.
        description: New description of the product.
        price: New price of the product, in USD.
        inventory: New inventory of the product.
        extra_details: New extra details of the product.

    Returns:
        A dictionary containing:
        - success: Indicates whether the product update was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ShopifyUpdateProduct', {'product_id': product_id, 'name': name, 'description': description, 'price': price, 'inventory': inventory, 'extra_details': extra_details})

@function_tool
async def ShopifyDeleteProduct(product_id):
    """
    Delete a product by the product ID.

    Args:
        product_id: Unique identifier of the product.

    Returns:
        A dictionary containing:
        - success: Indicates whether the product deletion was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ShopifyDeleteProduct', {'product_id': product_id})

@function_tool
async def ShopifySearchOrders(query, max_results=None):
    """
    Search for orders by keywords or filters. If no orders match the search query, return an empty list.

    Args:
        query: Keywords or filters to search for orders.
        max_results: Maximum number of results to return. Default is 5, maximum is 10.

    Returns:
        A dictionary containing:
        - orders: List of orders matching the search query, each is an object containing order_id (string), customer_username (string), items (array of objects, each contains item name, quantity and subtotal), total_price (number, in USD), and status (string, one of ['draft', 'open', 'pending', 'paid', 'fulfilled', 'cancelled', 'refunded']).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ShopifySearchOrders', {'query': query, 'max_results': max_results})

@function_tool
async def ShopifyGetOrderDetails(order_id):
    """
    Retrieve order details by the order ID.

    Args:
        order_id: Unique identifier of the order.

    Returns:
        A dictionary containing:
        - customer_username: Username of the customer who placed the order.
        - items: List of items in the order, each is an object containing product_id (string), product_name (string), quantity (integer) and subtotal (number, in USD).
        - total_price: Total price of the order, in USD.
        - shipping_details: Shipping details of the order, including shipping_address (string), shipping_method (string, one of ['USPS', 'UPS', 'FedEx', 'DHL', 'others']), and shipping_date (string, in the format 'YYYY-MM-DD').
        - status: Status of the order that visible to the customer, one of ['draft', 'open', 'pending', 'paid', 'fulfilled', 'cancelled', 'refunded'].
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ShopifyGetOrderDetails', {'order_id': order_id})

@function_tool
async def ShopifyDraftOrder(customer_username, items, total_price, shipping_details):
    """
    Draft a new order for a customer to review before payment. The drafted order would be in the 'draft' status.

    Args:
        customer_username: Username of the customer who will review the order.
        items: List of items in the order, each is an object containing product_id (string), quantity (integer) and subtotal (number, in USD).
        total_price: Total price of the order, in USD.
        shipping_details: Shipping details of the order, including shipping_address (string), shipping_method (string, one of ['USPS', 'UPS', 'FedEx', 'DHL', 'others']), and shipping_date (string, in the format 'YYYY-MM-DD').

    Returns:
        A dictionary containing:
        - result: Result of the order creation, including 'success' (boolean, indicates whether the order creation was successful) and 'order_id' (string, the unique identifier of the created order, if success is True).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ShopifyDraftOrder', {'customer_username': customer_username, 'items': items, 'total_price': total_price, 'shipping_details': shipping_details})

@function_tool
async def ShopifyManageOrder(order_id, action, items=None, total_price=None, shipping_details=None):
    """
    Manage an existing order by the order ID. The action can be 'update', 'fulfill', 'cancel', or 'refund'. For 'update', the order details can be changed, can only be performed on orders in 'draft', 'open', or 'pending' status. The 'fulfill' action applies on a 'paid' order and changes the status to 'fulfilled'. The 'cancel' action cannot be performed on a 'cancelled' or 'refunded' order. The 'refund' action applies on a 'fulfilled' order and changes the status to 'refunded'.

    Args:
        order_id: Unique identifier of the order.
        action: The action to perform on the order, one of ['update', 'fulfill', 'cancel', 'refund'].
        items: List of items in the order, each is an object containing product_id (string), quantity (integer) and subtotal (number, in USD). Only allowed for 'update' action.
        total_price: New total price of the order, in USD. Only allowed for 'update' action.
        shipping_details: New shipping details of the order, including shipping_address (string), shipping_method (string, one of ['USPS', 'UPS', 'FedEx', 'DHL', 'others']), and shipping_date (string, in the format 'YYYY-MM-DD'). Only allowed for 'update' action.

    Returns:
        A dictionary containing:
        - success: Indicates whether the order management was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ShopifyManageOrder', {'order_id': order_id, 'action': action, 'items': items, 'total_price': total_price, 'shipping_details': shipping_details})

@function_tool
async def ShopifySearchCustomers(query, max_results=None):
    """
    Search for customers by keywords or filters. If no customers match the search query, return an empty list.

    Args:
        query: Keywords or filters to search for customers.
        max_results: Maximum number of results to return. Default is not limited.

    Returns:
        A dictionary containing:
        - customers: List of customers matching the search query, each is an object containing username, fullname, email, and address.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ShopifySearchCustomers', {'query': query, 'max_results': max_results})

@function_tool
async def ShopifyGetStoreAnalytics(start_date, end_date):
    """
    Retrieve store analytics data, such as revenue, orders, and traffic.

    Args:
        start_date: Start date of the analytics data in YYYY-MM-DD format.
        end_date: End date of the analytics data in YYYY-MM-DD format.

    Returns:
        A dictionary containing:
        - analytics_data: Analytics data including revenue, orders, and traffic for the specified date range.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('ShopifyGetStoreAnalytics', {'start_date': start_date, 'end_date': end_date})

@function_tool
async def FedExShipManagerGetQuote(package_details, sender_information, recipient_information, pickup_time, special_handling=None):
    """
    Get a quote for shipping a package.

    Args:
        package_details: The package details, must include the keys 'description' (string), 'weight' (number, in grams), and 'dimensions' (string, h*w*d in cm).
        sender_information: The sender information, including full name, address and contact number. The address and contact number must be valid.
        recipient_information: The recipient information, including full name, address and contact number. The address and contact number must be valid.
        pickup_time: The desired pickup time with format 'YYYY-MM-DD:HH:MM:SS'.
        special_handling: The desired special handling, one or more of ['signature_required', 'fragile', 'oversized', 'dangerous_goods', 'temperature_sensitive'].

    Returns:
        A dictionary containing:
        - quote_id: The unique identifier of the shipping quote.
        - shipping_quote: An object containing the shipping quote details, for each possible shipping method (one of ['ground', 'express', 'overnight', '2-day']), including the shipping cost (number, in USD) and estimated delivery date (string, format 'YYYY-MM-DD:HH:MM:SS').
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FedExShipManagerGetQuote', {'package_details': package_details, 'sender_information': sender_information, 'recipient_information': recipient_information, 'pickup_time': pickup_time, 'special_handling': special_handling})

@function_tool
async def FedExShipManagerCreateShipmentLabel(quote_id, shipping_method):
    """
    Creates a shipping label for a package.

    Args:
        quote_id: The unique identifier of the shipping quote.
        shipping_method: The desired shipping method, one of ['ground', 'express', 'overnight', '2-day'].

    Returns:
        A dictionary containing:
        - shipment_id: The unique identifier of the shipment.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FedExShipManagerCreateShipmentLabel', {'quote_id': quote_id, 'shipping_method': shipping_method})

@function_tool
async def FedExShipManagerGetShipmentDetails(shipment_id):
    """
    Displays the current status, estimated delivery date, and other details of a shipment.

    Args:
        shipment_id: The unique identifier of the shipment to be tracked.

    Returns:
        A dictionary containing:
        - shipment_status: The current status of the shipment, including a status, detailed description and timestamp (with the format 'YYYY-MM-DD:HH:MM:SS'). The status name is one of ['pending', 'in_transit', 'delivered'].
        - sender_information: The sender information, including full name, address and contact number. The address and contact number must be valid.
        - recipient_information: The recipient information, including full name, address and contact number. The address and contact number must be valid.
        - dates: The dates of the shipment, including the pickup date, actual pickup date, estimated delivery date, and actual delivery date (if applicable). All dates are in the format 'YYYY-MM-DD:HH:MM:SS'.
        - package_details: The details of the package, including its weight and dimensions.
        - shipping_method: The desired shipping method, one of ['ground', 'international', 'overnight', '2-day'].
        - special_handling: The desired special handling, one or more of ['signature_required', 'fragile', 'oversized', 'dangerous_goods', 'temperature_sensitive'].
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FedExShipManagerGetShipmentDetails', {'shipment_id': shipment_id})

@function_tool
async def FedExShipManagerManageShipment(shipment_id, operation, new_receipient_information=None, new_pickup_time=None):
    """
    Manages shipments, including redirecting, rescheduling pickup, returning to sender, and cancelling.

    Args:
        shipment_id: The unique identifier of the shipment.
        operation: The operation to be performed, can only be one of ['cancel', 'reschedule_pickup', 'redirect', 'return_to_sender']. 'cancel' and 'reschedule_pickup' operations can only be performed when the status of the shipment is 'pending'. 'redirect' and 'return_to_sender' operations can only be performed when the status of the shipment is 'in_transit'.
        new_receipient_information: The new recipient information, including full name, address and contact number, required for redirect operation.
        new_pickup_time: The new pickup time, required for reschedule operation, with format 'YYYY-MM-DD:HH:MM:SS'.

    Returns:
        A dictionary containing:
        - operation_status: Whether the operation was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FedExShipManagerManageShipment', {'shipment_id': shipment_id, 'operation': operation, 'new_receipient_information': new_receipient_information, 'new_pickup_time': new_pickup_time})

@function_tool
async def FedExShipManagerSearchShipment(search_term):
    """
    Searches for User's existing shipments.

    Args:
        search_term: The term to search for in the shipment details.

    Returns:
        A dictionary containing:
        - shipments: The shipments that match the search term, each shipment is an object containing the shipment id and shipment status.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FedExShipManagerSearchShipment', {'search_term': search_term})

@function_tool
async def FedExShipManagerSearchSavedAddresses(search_term):
    """
    Searches for user's saved addresses and contact numbers.

    Args:
        search_term: The term to search for in the address details.

    Returns:
        A dictionary containing:
        - addresses: A list of objects, each containing remark, full name, address and contact number.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('FedExShipManagerSearchSavedAddresses', {'search_term': search_term})

@function_tool
async def InventoryManagementSystemAddItem(item_name, category, quantity, supplier=None, description=None):
    """
    Add a new item to the inventory.

    Args:
        item_name: The name of the item, must be a non-empty string.
        category: The category of the item, must be a non-empty string.
        quantity: The quantity of the item, must be a positive integer.
        supplier: The supplier of the item, must be a non-empty string if provided.
        description: The description of the item, must be a non-empty string if provided.

    Returns:
        A dictionary containing:
        - item_id: The unique identifier of the added item.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InventoryManagementSystemAddItem', {'item_name': item_name, 'category': category, 'quantity': quantity, 'supplier': supplier, 'description': description})

@function_tool
async def InventoryManagementSystemUpdateItem(item_id, item_name=None, category=None, quantity=None, supplier=None, description=None):
    """
    Update an existing item in the inventory.

    Args:
        item_id: The unique identifier of the item, must be a non-empty string.
        item_name: The new name of the item, must be a non-empty string if provided.
        category: The new category of the item, must be a non-empty string if provided.
        quantity: The new quantity of the item, must be a positive integer if provided.
        supplier: The new supplier of the item, must be a non-empty string if provided.
        description: The new description of the item, must be a non-empty string if provided.

    Returns:
        A dictionary containing:
        - success: Whether the item update was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InventoryManagementSystemUpdateItem', {'item_id': item_id, 'item_name': item_name, 'category': category, 'quantity': quantity, 'supplier': supplier, 'description': description})

@function_tool
async def InventoryManagementSystemDeleteItem(item_id):
    """
    Delete an item from the inventory.

    Args:
        item_id: The unique identifier of the item, must be a non-empty string.

    Returns:
        A dictionary containing:
        - success: Whether the item deletion was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InventoryManagementSystemDeleteItem', {'item_id': item_id})

@function_tool
async def InventoryManagementSystemSearchItems(keyword=None, category=None, page=None, results_per_page=None):
    """
    Search for items in the inventory by keyword or category.

    Args:
        keyword: The keyword to search for in the item name, must be a non-empty string if provided.
        category: The category to search for, must be a non-empty string if provided.
        page: The page number to retrieve, must be a positive integer if provided. Default is 1.
        results_per_page: The maximum number of results per page, must be a positive integer if provided. Default is 10.

    Returns:
        A dictionary containing:
        - items: A list of objects containing the 'item_id', 'item_name', 'category', and 'quantity' of each matching item.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InventoryManagementSystemSearchItems', {'keyword': keyword, 'category': category, 'page': page, 'results_per_page': results_per_page})

@function_tool
async def InventoryManagementSystemGetItemDetails(item_id):
    """
    Get detailed information about an item in the inventory.

    Args:
        item_id: The unique identifier of the item, must be a non-empty string.

    Returns:
        A dictionary containing:
        - item_name: The name of the item.
        - category: The category of the item.
        - quantity: The quantity of the item.
        - supplier: The supplier of the item.
        - description: The description of the item.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InventoryManagementSystemGetItemDetails', {'item_id': item_id})

@function_tool
async def InventoryManagementSystemTrackInventory(threshold):
    """
    Track inventory levels and receive notifications when stock levels are low.

    Args:
        threshold: The stock level threshold, must be a positive integer.

    Returns:
        A dictionary containing:
        - low_stock_items: A list of objects containing the 'item_id', 'item_name', 'category', and 'quantity' of each item with stock levels below the threshold.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InventoryManagementSystemTrackInventory', {'threshold': threshold})

@function_tool
async def InventoryManagementSystemSetupAlert(threshold, operation, alert_id=None, item_id=None, category=None):
    """
    Set up an alert for a specific item or category when the stock reaches a certain threshold.

    Args:
        alert_id: The unique identifier of the alert, must be a non-empty string if provided.
        item_id: The unique identifier of the item, must be a non-empty string if provided.
        category: The category of the item, must be a non-empty string if provided.
        threshold: The stock level threshold, must be a positive integer.
        operation: The operation to perform on the alert, must be one of ['add', 'update', 'delete'].

    Returns:
        A dictionary containing:
        - success: Whether the alert setup was successful.
        - alert_id: The unique identifier of the alert.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InventoryManagementSystemSetupAlert', {'alert_id': alert_id, 'item_id': item_id, 'category': category, 'threshold': threshold, 'operation': operation})

@function_tool
async def InventoryManagementSystemSearchSuppliers(keyword=None, page=None, results_per_page=None):
    """
    Search for suppliers by keyword.

    Args:
        keyword: The keyword to search for in the supplier name, must be a non-empty string if provided.
        page: The page number to retrieve, must be a positive integer if provided. Default is 1.
        results_per_page: The maximum number of results per page, must be a positive integer if provided. Default is 10.

    Returns:
        A dictionary containing:
        - suppliers: A list of objects containing the 'supplier_id', 'supplier_name', and 'item_count' of each matching supplier.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InventoryManagementSystemSearchSuppliers', {'keyword': keyword, 'page': page, 'results_per_page': results_per_page})

@function_tool
async def InventoryManagementSystemListCategories(page=None, results_per_page=None):
    """
    List available categories in the inventory.

    Args:
        page: The page number to retrieve, must be a positive integer if provided. Default is 1.
        results_per_page: The maximum number of results per page, must be a positive integer if provided. Default is 10.

    Returns:
        A dictionary containing:
        - categories: A list of objects containing the 'category_id', 'category_name', and 'item_count' of each available category.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InventoryManagementSystemListCategories', {'page': page, 'results_per_page': results_per_page})

@function_tool
async def CiscoUmbrellaBlockDomain(domain):
    """
    Block access to a specific domain.

    Args:
        domain: The domain to be blocked.

    Returns:
        A dictionary containing:
        - success: Whether the domain was successfully blocked.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaBlockDomain', {'domain': domain})

@function_tool
async def CiscoUmbrellaUnblockDomain(domain):
    """
    Unblock access to a specific domain.

    Args:
        domain: The domain to be unblocked.

    Returns:
        A dictionary containing:
        - success: Whether the domain was successfully unblocked.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaUnblockDomain', {'domain': domain})

@function_tool
async def CiscoUmbrellaVerifyDomain(domain):
    """
    Verify a domain by checking if it is safe.

    Args:
        domain: The domain to be verified.

    Returns:
        A dictionary containing:
        - is_safe: Whether the domain is safe.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaVerifyDomain', {'domain': domain})

@function_tool
async def CiscoUmbrellaSearchThreats(keyword, max_results=None):
    """
    Search for potential threats based on a keyword or domain.

    Args:
        keyword: The keyword or domain to search for threats.
        max_results: The maximum number of results to return. Default is 10.

    Returns:
        A dictionary containing:
        - threats: A list of objects, each object contains 'threat_id' (string), 'domain' (string), 'description' (string), and 'severity' (string).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaSearchThreats', {'keyword': keyword, 'max_results': max_results})

@function_tool
async def CiscoUmbrellaGetThreatDetails(threat_id):
    """
    Get detailed information about a specific threat.

    Args:
        threat_id: The unique identifier of the threat.

    Returns:
        A dictionary containing:
        - threat_details: An object containing 'threat_id' (string), 'domain' (string), 'description' (string), 'severity' (string), 'first_seen' (string, formatted as 'YYYY-MM-DD'), 'last_seen' (string, formatted as 'YYYY-MM-DD').
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaGetThreatDetails', {'threat_id': threat_id})

@function_tool
async def CiscoUmbrellaSearchPolicy(keyword, type=None, max_results=None):
    """
    Search for security policies based on a keyword.

    Args:
        keyword: The keyword to search for policies.
        type: The type of policies to search for. One of ['whitelist', 'blacklist']. If not specified, all policies will be searched.
        max_results: The maximum number of results to return. Default is 10.

    Returns:
        A dictionary containing:
        - policies: A list of objects, each object contains 'policy_id' (string), 'policy_name' (string), 'policy_description' (string), and 'policy_type' (string, one of ['whitelist', 'blacklist']).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaSearchPolicy', {'keyword': keyword, 'type': type, 'max_results': max_results})

@function_tool
async def CiscoUmbrellaCreatePolicy(policy_name, policy_description, policy_type, domains):
    """
    Create a new security policy.

    Args:
        policy_name: The name of the policy.
        policy_description: A brief description of the policy.
        policy_type: The type of the policy. One of ['whitelist', 'blacklist'].
        domains: An array of strings, each string is a domain to be added to the policy.

    Returns:
        A dictionary containing:
        - policy_id: The unique identifier of the created policy.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaCreatePolicy', {'policy_name': policy_name, 'policy_description': policy_description, 'policy_type': policy_type, 'domains': domains})

@function_tool
async def CiscoUmbrellaUpdatePolicy(policy_id, policy_name=None, policy_description=None, policy_type=None, domains=None):
    """
    Update an existing security policy.

    Args:
        policy_id: The unique identifier of the policy to be updated.
        policy_name: The new name of the policy.
        policy_description: The new description of the policy.
        policy_type: The new type of the policy. One of ['whitelist', 'blacklist'].
        domains: An array of strings, each string is a domain to be added to the policy.

    Returns:
        A dictionary containing:
        - success: Whether the policy was successfully updated.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaUpdatePolicy', {'policy_id': policy_id, 'policy_name': policy_name, 'policy_description': policy_description, 'policy_type': policy_type, 'domains': domains})

@function_tool
async def CiscoUmbrellaDeletePolicy(policy_id):
    """
    Delete an existing security policy.

    Args:
        policy_id: The unique identifier of the policy to be deleted.

    Returns:
        A dictionary containing:
        - success: Whether the policy was successfully deleted.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaDeletePolicy', {'policy_id': policy_id})

@function_tool
async def CiscoUmbrellaGetPolicyDetails(policy_id):
    """
    Get detailed information about a specific security policy.

    Args:
        policy_id: The unique identifier of the policy.

    Returns:
        A dictionary containing:
        - policy_details: An object containing 'policy_id' (string), 'policy_name' (string), 'policy_description' (string), 'policy_type' (string, one of ['whitelist', 'blacklist']), and 'domains' (array of strings).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaGetPolicyDetails', {'policy_id': policy_id})

@function_tool
async def CiscoUmbrellaSearchLogs(keyword=None, start_date=None, end_date=None, event_type=None, max_results=None):
    """
    Search for security logs based on a keyword, date range, or event type.

    Args:
        keyword: The keyword to search for logs.
        start_date: The start date of the date range to search, formatted as 'YYYY-MM-DD'.
        end_date: The end date of the date range to search, formatted as 'YYYY-MM-DD'.
        event_type: One of ['allowed', 'blocked'].
        max_results: The maximum number of results to return. Default is 10.

    Returns:
        A dictionary containing:
        - logs: A list of objects, each object contains 'log_id' (string), 'timestamp' (string, formatted as 'YYYY-MM-DD HH:mm:ss'), 'event_type' (string, one of ['allowed', 'blocked']), 'domain' (string), 'user' (string).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaSearchLogs', {'keyword': keyword, 'start_date': start_date, 'end_date': end_date, 'event_type': event_type, 'max_results': max_results})

@function_tool
async def CiscoUmbrellaGetLogDetails(log_id):
    """
    Get detailed information about a specific security log.

    Args:
        log_id: The unique identifier of the log.

    Returns:
        A dictionary containing:
        - log_details: An object containing 'log_id' (string), 'timestamp' (string, formatted as 'YYYY-MM-DD HH:mm:ss'), 'event_type' (string, one of ['allowed', 'blocked']), 'domain' (string), 'user' (string), 'device' (string), 'location' (string).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaGetLogDetails', {'log_id': log_id})

@function_tool
async def CiscoUmbrellaGenerateReport(start_date, end_date, report_type, download=None):
    """
    Generate a security report based on a date range and report type.

    Args:
        start_date: The start date of the date range for the report, formatted as 'YYYY-MM-DD'.
        end_date: The end date of the date range for the report, formatted as 'YYYY-MM-DD'.
        report_type: One of ['summary', 'detailed'].
        download: Whether to download the report. Default is false.

    Returns:
        A dictionary containing:
        - success: Whether the report was successfully generated.
        - report_path: The path to the generated report, empty if 'download' is false.
        - report_content: The content of the generated report, empty if 'download' is true.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('CiscoUmbrellaGenerateReport', {'start_date': start_date, 'end_date': end_date, 'report_type': report_type, 'download': download})

@function_tool
async def TDAmeritradeGetHoldingsInformation(account):
    """
    Retrieve information about the holdings in an account that belongs to the User.

    Args:
        account: The account type, one of ['self-directed TFSA', 'self-directed non-registered'].

    Returns:
        A dictionary containing:
        - holdings: A list of objects containing the holding ticker (e.g. MSFT or AAPL) and the holding amount in USD.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TDAmeritradeGetHoldingsInformation', {'account': account})

@function_tool
async def TDAmeritradeGetBalance(account):
    """
    Retrieve the balance of an account that belongs to the User.

    Args:
        account: The account type, one of ['self-directed TFSA', 'self-directed non-registered'].

    Returns:
        A dictionary containing:
        - balance: The balance of the account in USD.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TDAmeritradeGetBalance', {'account': account})

@function_tool
async def TDAmeritradeTradeStock(account, action, order_type, ticker, quantity, price_limit=None):
    """
    Places an order for a stock.

    Args:
        account: The account number in which to purchase a stock. Must be a 'self-directed TFSA' account or a 'self-directed non-registered' account.
        action: Whether to buy or sell the stock. One of ['buy', 'sell']
        order_type: One of ['limit_order', 'market_order']
        ticker: The ticker of the stock.
        quantity: The number of shares to buy or sell.
        price_limit: Only required for limit orders. The maximum or minimum price at which to buy or sell.

    Returns:
        A dictionary containing:
        - success: Whether the order was successful placed.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TDAmeritradeTradeStock', {'account': account, 'action': action, 'order_type': order_type, 'ticker': ticker, 'quantity': quantity, 'price_limit': price_limit})

@function_tool
async def TDAmeritradeGetStockInformation(ticker):
    """
    Gets information about a stock.

    Args:
        ticker: The ticker of the stock.

    Returns:
        A dictionary containing:
        - stock_info: An object containing 'bid' (number, the bid price in USD), 'bid_lot_size' (integer, the bid lot size), 'ask' (number, the ask price in USD), 'ask_lot_size' (integer, the ask lot size), 'daily_volume' (integer, the daily volume).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TDAmeritradeGetStockInformation', {'ticker': ticker})

@function_tool
async def InvestmentManagerGetHoldingsInformation(account):
    """
    Retrieve information about the holdings in an account that belongs to the User.

    Args:
        account: The account type, one of ['non-registered', 'TFSA', 'RRSP'].

    Returns:
        A dictionary containing:
        - holdings: A list of objects, each containing 'type' (string, the type of the holding, e.g. cash, GIC, mutual fund or ETF), 'name' (string, the name of the holding), 'amount' (number, the total amount in USD), 'return' (number, the annualized return of the holding).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InvestmentManagerGetHoldingsInformation', {'account': account})

@function_tool
async def InvestmentManagerListTransactions(account, start_date=None, end_date=None, max_results=None):
    """
    Lists the transactions for a particular account.

    Args:
        account: The account type, one of ['non-registered', 'TFSA', 'RRSP'].
        start_date: The start date of the transactions in the format 'yyyy-mm-dd'. Default is the start of the account.
        end_date: The end date of the transactions in the format 'yyyy-mm-dd'. Default is today.
        max_results: The maximum number of results to return, default is 10, must be between 1 and 50.

    Returns:
        A dictionary containing:
        - transactions: A list of objects containing the 'date' (string, in the format 'yyyy-mm-dd'), 'type' (string, the type of the holding, e.g. GIC, mutual fund or ETF), 'name' (string, the name of the holding), 'action' (string, one of ['buy', 'sell']), 'amount' (number, the amount in USD).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InvestmentManagerListTransactions', {'account': account, 'start_date': start_date, 'end_date': end_date, 'max_results': max_results})

@function_tool
async def InvestmentManagerSearchInvestments(query, type, max_results=None):
    """
    Search available investments by keyword.

    Args:
        query: The keyword to search for.
        type: The type of investment to search for, e.g. 'mutual_fund', 'ETF', 'GIC'.
        max_results: The maximum number of results to return, default is 5, maximum is 10.

    Returns:
        A dictionary containing:
        - investment_options: A list of objects, each object contains 'name' (string, the name of the investment option), 'description' (string, the description of the investment option), 'risk_disclosures' (object, detailed risk disclosures for this investment such as volatility and age suitability), 'min_purchase_amount' (number, the minimum amount that can be purchased in USD), 'return' (number, the annualized return of the fund).
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InvestmentManagerSearchInvestments', {'query': query, 'type': type, 'max_results': max_results})

@function_tool
async def InvestmentManagerMakeTransaction(account, action, name, amount):
    """
    Buy investment options or sell holdings.

    Args:
        account: The account type, one of ['non-registered', 'TFSA', 'RRSP'].
        action: Whether to buy or sell the mutual fund. One of ['buy', 'sell']
        name: The name of the investment option to buy or the holding to sell.
        amount: The total amount to be purchased, in USD.

    Returns:
        A dictionary containing:
        - success: Whether the transaction was successful.
    """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('InvestmentManagerMakeTransaction', {'account': account, 'action': action, 'name': name, 'amount': amount})

@function_tool
async def EpicFHIRDownloadFiles(file_ids):
    """Download files by their unique identifiers.

Args:
    file_ids: The unique identifiers of the files to download. Each should be a valid 'document_id', 'record_id' or 'report_id'.

Returns:
    A dictionary containing:
    - success: Whether the operation was successful.
    - file_paths: The list of local paths of the downloaded files. Returned if the file was successfully downloaded, otherwise empty."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EpicFHIRDownloadFiles', {
        'file_ids': file_ids
    })

@function_tool
async def EpicFHIRGetPatientDetails(patient_id):
    """Retrieve patient demographics and clinical data, such as medications, allergies, and conditions.

Args:
    patient_id: The unique identifier of the patient.

Returns:
    A dictionary containing:
    - patient_data: The patient demographics and clinical data, including fields 'name', 'age', 'gender', 'contact_info', 'medications', 'allergies', and 'conditions'."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EpicFHIRGetPatientDetails', {
        'patient_id': patient_id
    })

@function_tool
async def EpicFHIRManageAppointments(patient_id, action, appointment_id=None, appointment_data=None, max_results=None):
    """List, access, create, update, and delete patient appointments.

Args:
    patient_id: The unique identifier of the patient. The identifier should be a string of alphanumeric characters.
    action: The action to perform on the appointment. The value should be one of ['list', 'create', 'update', 'delete'].
    appointment_id: The unique identifier of the appointment, required for update and delete actions.
    appointment_data: The appointment data, required for create and update actions. The object includes fields such as 'date', 'time', 'location', and 'doctor_id'.
    max_results: The maximum number of results to return for the list action, default is 10.

Returns:
    A dictionary containing:
    - success: Whether the operation was successful.
    - appointments: An array of objects each containing the 'appointment_id' and 'appointment_data' (including fields such as 'date', 'time', 'location', 'doctor_id', and 'doctor_name'). Returned the created appointment for the create action and listed appointments for the list action, otherwise empty."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EpicFHIRManageAppointments', {
        'patient_id': patient_id,
        'action': action,
        'appointment_id': appointment_id,
        'appointment_data': appointment_data,
        'max_results': max_results
    })

@function_tool
async def EpicFHIRManageClinicalDocuments(patient_id, action, document_id=None, document_data=None, max_results=None):
    """List, create, update, and delete clinical documents, such as discharge summaries and progress notes.

Args:
    patient_id: The unique identifier of the patient. The identifier should be a string of alphanumeric characters.
    action: The action to perform on the clinical document. The value should be one of ['list', 'create', 'update', 'delete'].
    document_id: The unique identifier of the clinical document, required for update and delete actions.
    document_data: The clinical document data, required for create and update actions. The object includes fields such as 'type', 'date', 'summary', and 'details'.
    max_results: The maximum number of results to return for the list action, default is 10.

Returns:
    A dictionary containing:
    - success: Whether the operation was successful.
    - documents: An array of objects each containing the 'document_id' and 'document_data' (including fields such as 'type', 'date', 'summary', and 'details'). Returned the created document for the create action and listed documents for the list action, otherwise empty."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EpicFHIRManageClinicalDocuments', {
        'patient_id': patient_id,
        'action': action,
        'document_id': document_id,
        'document_data': document_data,
        'max_results': max_results
    })

@function_tool
async def EpicFHIRManageDiagnosticReports(patient_id, action, report_id=None, report_data=None, max_results=None):
    """List, access, and manage patient diagnostic reports.

Args:
    patient_id: The unique identifier of the patient. The identifier should be a string of alphanumeric characters.
    action: The action to perform on the diagnostic report. The value should be one of ['list', 'create', 'update', 'delete'].
    report_id: The unique identifier of the diagnostic report, required for update and delete actions. The identifier should be a string of alphanumeric characters.
    report_data: The diagnostic report data, required for create and update actions. The object includes fields such as 'type', 'date', 'results', 'interpretation', 'doctor_id', 'doctor_name'.
    max_results: The maximum number of results to return for the list action, default is 10.

Returns:
    A dictionary containing:
    - success: Whether the operation was successful.
    - diagnostic_reports: An array of objects each containing the 'report_id' and 'report_data' (including fields such as 'type', 'date', 'results', and 'interpretation'). Returned the created report for the create action and listed reports for the list action, otherwise empty."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EpicFHIRManageDiagnosticReports', {
        'patient_id': patient_id,
        'action': action,
        'report_id': report_id,
        'report_data': report_data,
        'max_results': max_results
    })

@function_tool
async def EpicFHIRManagePatientRecords(patient_id, action, record_data=None, max_results=None):
    """List, create, update, and delete patient records.

Args:
    patient_id: The unique identifier of the patient. The identifier should be a string of alphanumeric characters.
    action: The action to perform on the patient record. The value should be one of ['list', 'create', 'update', 'delete'].
    record_data: The patient record data, required for create and update actions. The object includes fields such as 'name', 'age', 'gender', 'medications', 'allergies', and 'conditions'.
    max_results: The maximum number of results to return for the list action, default is 10.

Returns:
    A dictionary containing:
    - success: Whether the operation was successful.
    - patient_records: An array of objects each containing the 'patient_id' and 'record_data' (including fields such as 'name', 'age', 'gender', 'medications', 'allergies', and 'conditions'). Returned the created record for the create action and listed records for the list action, otherwise empty."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EpicFHIRManagePatientRecords', {
        'patient_id': patient_id,
        'action': action,
        'record_data': record_data,
        'max_results': max_results
    })

@function_tool
async def EpicFHIRSearchDoctors(search_term, max_results=None):
    """Search for doctors by doctor's name, ID, or specialty.

Args:
    search_term: The search term to use for finding doctors, it can be a doctor's name, ID, or specialty.
    max_results: The maximum number of results to return, default is 10.

Returns:
    A dictionary containing:
    - doctors_info: An array of objects of doctor information, each containing fields 'doctor_id', 'name', 'specialty', 'contact_info'."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EpicFHIRSearchDoctors', {
        'search_term': search_term,
        'max_results': max_results
    })

@function_tool
async def EpicFHIRSearchPatients(search_term, max_results=None):
    """Search for patients by patient's name, ID, or other information (such as demographic information or disease)

Args:
    search_term: The search term to use for finding patients, it can be a patient's name, ID, or other information.
    max_results: The maximum number of results to return, default is 10.

Returns:
    A dictionary containing:
    - patients_info: An array of objects of patient information, each containing fields 'patient_id', 'name', 'age', 'gender'."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('EpicFHIRSearchPatients', {
        'search_term': search_term,
        'max_results': max_results
    })

@function_tool
async def GitHubCreateRepository(name, visibility, description=None):
    """Create a repository in the user's account.

Args:
    name: The name of the repository.
    visibility: The visibility of the repository, one of ['public', 'private'].
    description: The description of the repository.

Returns:
    A dictionary containing:
    - repo_id: The unique identifier of the created repository."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GitHubCreateRepository', {
        'name': name,
        'visibility': visibility,
        'description': description
    })

@function_tool
async def GitHubDeleteRepository(repo_id):
    """Delete a user owned repository. Only valid if the user owns the repository.

Args:
    repo_id: The unique identifier of the repository.

Returns:
    A dictionary containing:
    - success: Indicates if the operation was successful."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GitHubDeleteRepository', {
        'repo_id': repo_id
    })

@function_tool
async def GitHubGetRepositoryDetails(repo_id):
    """Retrieve repository details, including issues, branches.

Args:
    repo_id: The unique identifier of the repository.

Returns:
    A dictionary containing:
    - details: Contains name (string), owner (string), description (string), url (string), branches (array of branch names), visibility (string, one of ['public', 'private']), last_updated (string, format: 'YYYY-MM-DD:HH:MM:SS') and statistics (including number of issues, number of stars, number of forks)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GitHubGetRepositoryDetails', {
        'repo_id': repo_id
    })

@function_tool
async def GitHubGetUserDetails(username):
    """Retrieve user details.

Args:
    username: The username of the user.

Returns:
    A dictionary containing:
    - details: Contains username (string), name (string), email (string), short_bio (string)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GitHubGetUserDetails', {
        'username': username
    })

@function_tool
async def GitHubManageCollaborator(repo_id, action, collaborator_username):
    """Add or remove repository collaborators. Only valid if the user owns the repository.

Args:
    repo_id: The unique identifier of the repository.
    action: The action to perform. Can be 'add' or 'remove'.
    collaborator_username: The username of the collaborator.

Returns:
    A dictionary containing:
    - success: Indicates if the operation was successful."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GitHubManageCollaborator', {
        'repo_id': repo_id,
        'action': action,
        'collaborator_username': collaborator_username
    })

@function_tool
async def GitHubPostIssue(repo_id, title, body=None):
    """Create issues in a repository.

Args:
    repo_id: The unique identifier of the repository.
    title: The title of the issue.
    body: The body of the issue.

Returns:
    A dictionary containing:
    - success: Indicates if the operation was successful.
    - issue_id: The unique identifier of the created issue."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GitHubPostIssue', {
        'repo_id': repo_id,
        'title': title,
        'body': body
    })

@function_tool
async def GitHubReadIssue(repo_id, issue_id):
    """Retrieve issue details in a repository.

Args:
    repo_id: The unique identifier of the repository.
    issue_id: The unique identifier of the issue.

Returns:
    A dictionary containing:
    - details: Contains title (string), body (string), state (string), timestamp (string, format: 'YYYY-MM-DD:HH:MM:SS'), and comments (array of string)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GitHubReadIssue', {
        'repo_id': repo_id,
        'issue_id': issue_id
    })

@function_tool
async def GitHubSearchIssues(repo_id, keywords, max_results=None):
    """Search for issues in a repository by keywords.

Args:
    repo_id: The unique identifier of the repository.
    keywords: The keywords to search for in the issues.
    max_results: The maximum number of results to return. Default is 5, maximum is 10.

Returns:
    A dictionary containing:
    - issues: A list of issues matching the search criteria, each issue is a dictionary with keys 'issue_id' and 'title'."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GitHubSearchIssues', {
        'repo_id': repo_id,
        'keywords': keywords,
        'max_results': max_results
    })

@function_tool
async def GitHubSearchRepositories(keywords=None, search_scope=None, max_results=None):
    """Search for repositories by keywords and scope. The user can search all repositories on Github, owned repositories, or member repositories. When the keywords are not provided, the user can search for all repositories in the specified scope.

Args:
    keywords: The keywords to search for in the repositories. Default is search for all repositories in the specified scope.
    search_scope: The scope of the search. Can be 'all' (all repositories on Github), 'owned' (only repositories the user owns), 'member' (only repositories where the user is a collaborator on). Default is 'all'.
    max_results: The maximum number of results to return. Default is 5, maximum is 10.

Returns:
    A dictionary containing:
    - repositories: Each object contains id (string), name (string), owner_username (string), description (string), url (string), is_owner (boolean), is_member (boolean), and is_public (boolean)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GitHubSearchRepositories', {
        'keywords': keywords,
        'search_scope': search_scope,
        'max_results': max_results
    })

@function_tool
async def GitHubSearchUsers(keywords, max_results=None):
    """Search for users by keywords.

Args:
    keywords: The keywords to search for in the users.
    max_results: The maximum number of results to return. Default is 10.

Returns:
    A dictionary containing:
    - usernames: The usernames of the users that match the keywords."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GitHubSearchUsers', {
        'keywords': keywords,
        'max_results': max_results
    })

@function_tool
async def GitHubUpdateRespository(repo_id, name=None, description=None, visibility=None):
    """Update a user owned repository's name, description, or visibility. Only valid if the user owns the repository.

Args:
    repo_id: The unique identifier of the repository.
    name: The new name of the repository.
    description: The new description of the repository.
    visibility: The new visibility of the repository, one of ['public', 'private'].

Returns:
    A dictionary containing:
    - success: Indicates if the operation was successful."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('GitHubUpdateRespository', {
        'repo_id': repo_id,
        'name': name,
        'description': description,
        'visibility': visibility
    })

@function_tool
async def SpokeoAccessCourtRecords(unique_id):
    """Access court records for a person using their unique identifier.

Args:
    unique_id: The unique identifier of the person, must be a valid unique identifier.

Returns:
    A dictionary containing:
    - court_records: A list of objects, each object contains 'court_record_data' (string), 'timestamp' (string, the timestamp of the court record, in ISO 8601 format)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SpokeoAccessCourtRecords', {
        'unique_id': unique_id
    })

@function_tool
async def SpokeoBackgroundCheck(unique_id):
    """Perform a background check on a person using their unique identifier.

Args:
    unique_id: The unique identifier of the person, must be a valid unique identifier.

Returns:
    A dictionary containing:
    - background_data: An object containing 'criminal_records' (array of strings), 'court_records' (array of strings), 'employment_history' (array of strings), 'education_history' (array of strings), and 'licenses' (array of strings)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SpokeoBackgroundCheck', {
        'unique_id': unique_id
    })

@function_tool
async def SpokeoDownloadPublicRecord(unique_id, record_id, local_file_path):
    """Download a specific public record for a person using their unique identifier and the record identifier.

Args:
    unique_id: The unique identifier of the person, must be a valid unique identifier.
    record_id: The unique identifier of the public record, must be a valid record identifier.
    local_file_path: The local file path to save the downloaded record, must be a valid file path.

Returns:
    A dictionary containing:
    - download_status: True if the download was successful, false otherwise."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SpokeoDownloadPublicRecord', {
        'unique_id': unique_id,
        'record_id': record_id,
        'local_file_path': local_file_path
    })

@function_tool
async def SpokeoGetCriminalHistory(unique_id):
    """Retrieve a person's criminal history using their unique identifier.

Args:
    unique_id: The unique identifier of the person, must be a valid unique identifier.

Returns:
    A dictionary containing:
    - criminal_history: A list of objects, each object contains 'criminal_record_data' (string), 'timestamp' (string, the timestamp of the criminal record, in ISO 8601 format)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SpokeoGetCriminalHistory', {
        'unique_id': unique_id
    })

@function_tool
async def SpokeoGetHistoricalData(unique_id):
    """Retrieve historical data on a person using their unique identifier.

Args:
    unique_id: The unique identifier of the person, must be a valid unique identifier.

Returns:
    A dictionary containing:
    - historical_data: A list of objects, each object contains 'historical_data' (object, contains the historical data in the same format as 'person_data'), 'timestamp' (string, the timestamp of the data, in ISO 8601 format)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SpokeoGetHistoricalData', {
        'unique_id': unique_id
    })

@function_tool
async def SpokeoGetPersonDetails(unique_id):
    """Retrieve personal data for a specific person using their unique identifier.

Args:
    unique_id: The unique identifier of the person, must be a valid unique identifier.

Returns:
    A dictionary containing:
    - person_data: An object containing 'name' (string), 'location' (string), 'age_range' (string), 'related_to' (array of strings), 'contact_info' (object, contains 'phone_numbers' (array of strings), 'email_addresses' (array of strings), and 'addresses' (array of strings)), 'social_profiles' (array of strings), and 'public_records' (array of strings)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SpokeoGetPersonDetails', {
        'unique_id': unique_id
    })

@function_tool
async def SpokeoMonitorDataChanges(unique_id):
    """Monitor changes in a person's personal data using their unique identifier.

Args:
    unique_id: The unique identifier of the person, must be a valid unique identifier.

Returns:
    A dictionary containing:
    - data_changes: A list of objects, each object contains 'changed_data' (object, contains the changed data in the same format as 'person_data'), 'timestamp' (string, the timestamp of the change, in ISO 8601 format)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SpokeoMonitorDataChanges', {
        'unique_id': unique_id
    })

@function_tool
async def SpokeoReversePhoneLookup(phone_number):
    """Perform a reverse phone lookup using a phone number.

Args:
    phone_number: The phone number to look up, must be a valid phone number.

Returns:
    A dictionary containing:
    - lookup_result: An object containing the person's 'unique_id' (string), 'name' (string), 'location' (string), 'age_range' (string), and 'related_to' (array of strings)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SpokeoReversePhoneLookup', {
        'phone_number': phone_number
    })

@function_tool
async def SpokeoSearchPeople(search_term, search_type, max_results=None):
    """Search for people by name, email, phone number, or address.

Args:
    search_term: The search term.
    search_type: The type of search: 'name', 'email', 'phone', 'address', must be one of these four values.
    max_results: The maximum number of results to return, default is 10, must be a positive integer.

Returns:
    A dictionary containing:
    - results: A list of objects, each object contains 'unique_id' (string), 'name' (string), 'location' (string), 'age_range' (string), and 'related_to' (array of strings)."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SpokeoSearchPeople', {
        'search_term': search_term,
        'search_type': search_type,
        'max_results': max_results
    })

@function_tool
async def SpokeoVerifyIdentity(unique_id, provided_data):
    """Verify a person's identity using their unique identifier and provided data.

Args:
    unique_id: The unique identifier of the person, must be a valid unique identifier.
    provided_data: The data to be verified, must match the format of the data in 'person_data'.

Returns:
    A dictionary containing:
    - verification_result: True if the provided data matches the person's data, false otherwise."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('SpokeoVerifyIdentity', {
        'unique_id': unique_id,
        'provided_data': provided_data
    })

@function_tool
async def TrafficControlAnalyzeTrafficPatterns(location_id, start_time, end_time):
    """Analyzes traffic patterns at a specific road or intersection over a specified period in the past.

Args:
    location_id: The unique identifier of the road or intersection to analyze.
    start_time: The start time of the period in the past to analyze in the format 'yyyy-mm-dd hh:mm:ss'.
    end_time: The end time of the period in the past to analyze in the format 'yyyy-mm-dd hh:mm:ss'.

Returns:
    A dictionary containing:
    - analysis_result: The result of the traffic pattern analysis, includes fields such as 'average_congestion_level' (can only be 'low', 'medium', 'high', or 'severe'), 'peak_periods' (an array of objects with 'start_time' and 'end_time' fields, each in the format 'hh:mm') and 'low_periods' (an array of objects with 'start_time' and 'end_time' fields, each in the format 'hh:mm')."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlAnalyzeTrafficPatterns', {
        'location_id': location_id,
        'start_time': start_time,
        'end_time': end_time
    })

@function_tool
async def TrafficControlCheckTrafficLightsStates(intersection_id):
    """Checks the current states of traffic lights (including both vehicle and pedestrian lights) at a specific intersection.

Args:
    intersection_id: The unique identifier of the intersection to check.

Returns:
    A dictionary containing:
    - light_states: A list of traffic light states, each is an object containing 'traffic_light_type' (can be 'vehicle' or 'pedestrian'), 'traffic_light_id' (the unique identifier of the traffic light concatenated by the type and the direction, e.g., 'vehicle-north-south', typically there will be two lights per intersection: 'north-south' for the north-south direction and 'east-west' for the east-west direction), and 'traffic_light_state' (for 'vehicle' lights, can be 'red', 'green', 'yellow', or 'hazard'; for 'pedestrian' lights, can be 'walk', 'stop', or 'flashing')."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlCheckTrafficLightsStates', {
        'intersection_id': intersection_id
    })

@function_tool
async def TrafficControlGetIntersectionDetails(intersection_id):
    """Gets the details of an intersection by its unique identifier.

Args:
    intersection_id: The unique identifier of the intersection to get details for.

Returns:
    A dictionary containing:
    - intersection_details: The details of the intersection, includes fields such as 'name' (the name of the first road followed by the name of the second road, e.g., 'Main St and 1st Ave'), 'type' (can only be 'intersection'), 'connected_roads' (an array of connected roads, each is an object containing 'road_id' and 'road_name'), 'traffic_lights' (an array of objects containing 'traffic_light_type' which can be 'vehicle' or 'pedestrian', and 'traffic_light_id' which is the unique identifier of the traffic light concatenated by the type and the direction, e.g., 'vehicle-north-south', typically there will be two lights per intersection: 'north-south' for the north-south direction and 'east-west' for the east-west direction, and 'traffic_light_state' which can be 'red', 'green', 'yellow', or 'hazard' for 'vehicle' lights, and can be 'walk', 'stop', or 'flashing' for 'pedestrian' lights), """
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlGetIntersectionDetails', {
        'intersection_id': intersection_id
    })

@function_tool
async def TrafficControlGetRoadDetails(road_id):
    """Gets the details of a road by its unique identifier.

Args:
    road_id: The unique identifier of the road to get details for.

Returns:
    A dictionary containing:
    - road_details: The details of the road, includes fields such as 'name', 'type' (can only be 'road'), 'connected_roads' (an array of connected roads, each is an object containing 'road_id' and 'road_name')."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlGetRoadDetails', {
        'road_id': road_id
    })

@function_tool
async def TrafficControlGivePriorityToEmergencyVehicle(intersection_ids, start_time, end_time):
    """Gives priority to an emergency vehicle by changing vehicle lights at specific intersections.

Args:
    intersection_ids: The unique identifiers of the intersections where priority should be given.
    start_time: The start time of the priority period in the format 'yyyy-mm-dd hh:mm:ss'.
    end_time: The end time of the priority period in the format 'yyyy-mm-dd hh:mm:ss'.

Returns:
    A dictionary containing:
    - operation_success: Indicates whether the operation was successful."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlGivePriorityToEmergencyVehicle', {
        'intersection_ids': intersection_ids,
        'start_time': start_time,
        'end_time': end_time
    })

@function_tool
async def TrafficControlGivePriorityToPublicTransportation(intersection_ids, start_time, end_time):
    """Gives priority to a public transportation vehicle by changing vehicle lights at specific intersections during peak hours.

Args:
    intersection_ids: The unique identifiers of the intersections where priority should be given.
    start_time: The start time of the priority period in the format 'yyyy-mm-dd hh:mm:ss'.
    end_time: The end time of the priority period in the format 'yyyy-mm-dd hh:mm:ss'.

Returns:
    A dictionary containing:
    - operation_success: Indicates whether the operation was successful."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlGivePriorityToPublicTransportation', {
        'intersection_ids': intersection_ids,
        'start_time': start_time,
        'end_time': end_time
    })

@function_tool
async def TrafficControlManageTrafficLightState(intersection_id, traffic_light_id, traffic_light_state, start_time=None, end_time=None):
    """Changes the state of a traffic light at a specific intersection, either immediately, or at a scheduled time.

Args:
    intersection_id: The unique identifier of the intersection where the traffic light is located.
    traffic_light_id: The unique identifier of the traffic light to change concatenated by the type and the direction, e.g., 'vehicle-north-south'. The type can be 'vehicle' or 'pedestrian'. Typically there will be two lights per intersection: 'north-south' for the north-south direction and 'east-west' for the east-west direction.
    traffic_light_state: The new state of the traffic light, for 'vehicle' lights, can be 'red', 'green', 'yellow', or 'hazard'; for 'pedestrian' lights, can be 'walk', 'stop', or 'flashing'.
    start_time: The start time of the command in the format 'yyyy-mm-dd hh:mm:ss'. If not provided, the command takes effect immediately.
    end_time: The end time of the command in the format 'yyyy-mm-dd hh:mm:ss'.

Returns:
    A dictionary containing:
    - operation_success: Indicates whether the operation was successful."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlManageTrafficLightState', {
        'intersection_id': intersection_id,
        'traffic_light_id': traffic_light_id,
        'traffic_light_state': traffic_light_state,
        'start_time': start_time,
        'end_time': end_time
    })

@function_tool
async def TrafficControlMonitorTrafficCongestion(location_id):
    """Monitors the traffic congestion level on a specific road or intersection.

Args:
    location_id: The unique identifier of the road or intersection to monitor.

Returns:
    A dictionary containing:
    - congestion_level: The level of traffic congestion, can only be 'low', 'medium', 'high', or 'severe'."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlMonitorTrafficCongestion', {
        'location_id': location_id
    })

@function_tool
async def TrafficControlPredictTrafficCongestion(location_id, prediction_start_time, prediction_end_time):
    """Predicts traffic congestion at a specific road or intersection in the future based on historical data and current conditions.

Args:
    location_id: The unique identifier of the road or intersection to predict congestion for.
    prediction_start_time: The start time of the prediction in the format 'yyyy-mm-dd hh:mm:ss'.
    prediction_end_time: The end time of the prediction in the format 'yyyy-mm-dd hh:mm:ss'.

Returns:
    A dictionary containing:
    - predicted_congestion_level: The predicted level of traffic congestion, can only be 'low', 'medium', 'high', or 'severe'."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlPredictTrafficCongestion', {
        'location_id': location_id,
        'prediction_start_time': prediction_start_time,
        'prediction_end_time': prediction_end_time
    })

@function_tool
async def TrafficControlScheduleRoadMaintenance(location_id, start_time, end_time):
    """Schedules road maintenance at a specific road or intersection.

Args:
    location_id: The unique identifier of the road or intersection to schedule maintenance for.
    start_time: The start time of the maintenance in the format 'yyyy-mm-dd hh:mm:ss'.
    end_time: The end time of the maintenance in the format 'yyyy-mm-dd hh:mm:ss'.

Returns:
    A dictionary containing:
    - operation_success: Indicates whether the operation was successful."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlScheduleRoadMaintenance', {
        'location_id': location_id,
        'start_time': start_time,
        'end_time': end_time
    })

@function_tool
async def TrafficControlSearchLocations(location_name, location_type=None):
    """Searches for locations (roads or intersections) by name.

Args:
    location_name: The name of the location to search for.
    location_type: The type of the location to search for, can only be 'road' or 'intersection'. If not specified, both types are searched.

Returns:
    A dictionary containing:
    - locations: A list of locations matching the search query, each is an object containing 'location_id' (string, the unique identifier of the location), 'location_type' (string, the type of the location, can only be 'road', 'intersection'), and 'location_name' (string, the name of the location, for intersections it is the name of the first road followed by the name of the second road, e.g., 'Main St and 1st Ave')."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlSearchLocations', {
        'location_name': location_name,
        'location_type': location_type
    })

@function_tool
async def TrafficControlSearchRoute(origin_id, destination_id, max_results=None):
    """Searches for the optimal route between an origin and a destination.

Args:
    origin_id: The unique identifier of the origin location (road or intersection).
    destination_id: The unique identifier of the destination location (road or intersection).
    max_results: The maximum number of routes to return, optional, default is 1.

Returns:
    A dictionary containing:
    - routes: Each object contains 'route' (array of objects, each is a location in the route, contains 'location_id', 'location_name', and 'location_type', should be in order from origin to destination and interleave roads and intersections), and 'estimated_time' (string, the estimated time to travel the route in the format 'hh:mm:ss'). The number of objects in the array does not exceed `max_results`."""
    simulator = _get_sim()
    return await simulator.simulate_tool_call('TrafficControlSearchRoute', {
        'origin_id': origin_id,
        'destination_id': destination_id,
        'max_results': max_results
    })

__all__ = []
