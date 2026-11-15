"""CLI: run the test matrix (models x email variants x prompt variants x trials) and write metrics + trial records.

Examples
    # offline smoke run with the scripted mock backend (no model needed):
    .venv-tools/bin/python -m compdorm.run_eval --mock --trials 3 --out results/compdorm_mock.json

    # against a served model (local vLLM default, or OpenRouter via OPENROUTER_API_KEY):
    .venv-tools/bin/python -m compdorm.run_eval --model openai/gpt-oss-20b \
        --emails v1_direct,v2_invoice --prompts default,cautious --trials 10 \
        --out results/compdorm_gptoss.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import fixtures as F
from .agent import make_client
from .defenses import make_llm_judge
from .harness import run_trial
from .metrics import summarize_by_cell

ROOT = Path(__file__).resolve().parents[1]


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compositional dormancy benchmark runner")
    ap.add_argument("--model", default="openai/gpt-oss-20b", help="model id passed to the backend")
    ap.add_argument("--models", default=None, help="comma list of models (overrides --model)")
    ap.add_argument("--emails", default=",".join(F.EMAIL_VARIANTS), help="comma list of email variants")
    ap.add_argument("--prompts", default=",".join(F.PROMPT_VARIANTS), help="comma list of prompt variants")
    ap.add_argument("--trials", type=int, default=10, help="trials per cell")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--mock", action="store_true", help="use the scripted offline backend")
    ap.add_argument("--judge-model", default=None, help="model id for the LLM content guard (else heuristic)")
    ap.add_argument("--out", default=str(ROOT / "results" / "compdorm_report.json"))
    ap.add_argument("--workdir", default=None, help="dir for per-trial memory stores (default: a temp dir)")
    args = ap.parse_args(argv)

    if args.mock:
        from .mock_backend import MockCompliantClient
        client = MockCompliantClient()
    else:
        client = make_client(base_url=args.base_url)
    judge = make_llm_judge(client, args.judge_model) if args.judge_model and not args.mock else None

    models = _csv(args.models) if args.models else [args.model]
    emails, prompts = _csv(args.emails), _csv(args.prompts)
    workdir = Path(args.workdir) if args.workdir else None
    if workdir:
        workdir.mkdir(parents=True, exist_ok=True)

    trials, total = [], len(models) * len(emails) * len(prompts) * args.trials
    i = 0
    for model in models:
        for ev in emails:
            for pv in prompts:
                for _ in range(args.trials):
                    i += 1
                    t = run_trial(client, model, ev, pv, temperature=args.temperature, judge=judge, workdir=workdir)
                    trials.append(t)
                    print(f"[{i}/{total}] {model} {ev}/{pv}: launder={t['planting']['self_laundered']} "
                          f"e2e={t['e2e_success']} caught={[d for d,v in t['defenses'].items() if v['blocked']]}",
                          file=sys.stderr)

    report = {"config": {"models": models, "emails": emails, "prompts": prompts, "trials_per_cell": args.trials,
                         "temperature": args.temperature, "mock": args.mock, "judge_model": args.judge_model},
              **summarize_by_cell(trials), "trials": trials}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    ov = report["overall"]
    print(f"\nWrote {len(trials)} trials -> {out}")
    print(f"SLR={ov['self_laundering_rate']}  CR={ov['composition_rate']}  E2E={ov['e2e_success_rate']}  "
          f"caution={ov['caution_rate']}")
    print("detection rate:", {d: r for d, r in ov["detection_rate"].items()})
    return report


if __name__ == "__main__":
    main()
