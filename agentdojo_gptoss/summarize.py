"""Summarize AgentDojo run logs: python summarize.py [logdir=runs_t0] [pipeline=gpt-oss-20b]"""
import json, glob, sys, collections
logdir = sys.argv[1] if len(sys.argv) > 1 else "runs_t0"; pipe = sys.argv[2] if len(sys.argv) > 2 else "gpt-oss-20b"
rows = [json.load(open(f)) for f in glob.glob(f"{logdir}/{pipe}/*/*/*/*.json")]
def pct(xs): return f"{100*sum(xs)/len(xs):5.1f}% ({sum(xs)}/{len(xs)})" if xs else "  n/a"
def flags(d):
    ms = d["messages"]
    return dict(retried=any("retry_attempt" in m for m in ms), bad_args=any("tool_call_problems" in m for m in ms),
                server_error=any("server_errors" in m for m in ms), n_assistant=sum(m["role"]=="assistant" for m in ms))
by = collections.defaultdict(list)
for d in rows: by[(d["suite_name"], d["attack_type"])].append(d)
for (suite, attack), ds in sorted(by.items(), key=str):
    print(f"\n== suite={suite} attack={attack} episodes={len(ds)}")
    print("  utility            :", pct([d["utility"] for d in ds]))
    if attack:
        print("  attack success rate:", pct([d["security"] for d in ds]))
        per = collections.defaultdict(list)
        for d in ds: per[d["injection_task_id"]].append(d["security"])
        for k in sorted(per): print(f"    {k:18s} ASR {pct(per[k])}")
    fl = [flags(d) for d in ds]
    print(f"  episodes w/ server-error retries: {sum(f['retried'] for f in fl)}, w/ non-JSON tool args: {sum(f['bad_args'] for f in fl)}, "
          f"gave up (server error): {sum(f['server_error'] for f in fl)}, mean assistant turns: {sum(f['n_assistant'] for f in fl)/len(fl):.1f}")
