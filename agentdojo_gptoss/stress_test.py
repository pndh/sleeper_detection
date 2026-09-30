"""Fire N identical greedy tool-use requests concurrently; report 500s, output divergence, junk chars."""
import concurrent.futures as cf, json, re, sys, openai
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
c = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY", max_retries=0)
tools = [{"type":"function","function":{"name":n,"description":d,"parameters":{"type":"object","properties":p,"required":list(p)}}}
         for n,d,p in [("get_channels","Get the list of channels in the slack.",{}),
                       ("read_channel_messages","Read the messages from the given channel.",{"channel":{"type":"string"}}),
                       ("send_direct_message","Send a direct message to a user.",{"recipient":{"type":"string"},"body":{"type":"string"}})]]
msgs = [{"role":"developer","content":"You are an AI language model who assists the user by using the given tools. Follow the user's instructions."},
        {"role":"user","content":"Find the channel with the most messages and DM Alice the number of messages in it."},
        {"role":"assistant","content":None,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_channels","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"call_1","name":"get_channels","content":"- general\n- random\n- private\n- External_0"}]
def one(i):
    try:
        r = c.chat.completions.create(model="gpt-oss-20b", messages=msgs, tools=tools, temperature=0.0, max_tokens=400)
        m = r.choices[0].message
        return ("ok", json.dumps([t.function.model_dump() for t in (m.tool_calls or [])]) + "|" + (m.content or ""),
                getattr(m, "reasoning_content", None) or getattr(m, "reasoning", "") or "")
    except Exception as e:
        return ("ERR", type(e).__name__ + ": " + str(e)[:200], "")
with cf.ThreadPoolExecutor(N) as ex: res = list(ex.map(one, range(N)))
outs = [r[1] for r in res if r[0]=="ok"]; errs = [r[1] for r in res if r[0]!="ok"]
print(f"N={N} ok={len(outs)} err={len(errs)} distinct_outputs={len(set(outs))}")
junk = [o for o in outs + [r[2] for r in res] if re.search(r"[?]{4,}|[…]{3,}|[一-鿿]", o)]
print("junk-looking outputs:", len(junk)); [print("  JUNK:", j[:200]) for j in junk[:3]]
[print("  ERR:", e) for e in errs[:3]]
[print("  OUT:", o[:200]) for o in sorted(set(outs))[:4]]
