# Plan: public demo site for the dormant-payload detector

Implementer: an AI coding agent working in `/home/huynp2/sleeper_detection` on host `jupyter-huynp2`.
Owner: the researcher. Ask the owner whenever this plan says **ASK**. Do not guess on those.

## 0. What we are building and why

The detector sits on the API provider's side. It reads the model's own activations on an incoming request and flags
requests whose memory contains a planted instruction that is waiting to fire. To demo it you need the model running
locally with access to its internals. Nothing executes for real: tools are stubs, so a tool call is only text shown on
screen. That is what "sandboxed" means here.

Two deliverables, shipped in this order:

| Phase | What | Needs GPU | Audience |
|---|---|---|---|
| **A. Static site** | Landing page, precomputed examples, results chart, honest status. Public link. | No | Social post, email to friends |
| **B. Live sandbox** | Same page plus a "try it" panel. A backend runs gpt-oss-20b and the probe on a request the visitor edits. | Yes (fits in 20 GB) | Scheduled demo calls only |

Phase A has to be done and shareable even if Phase B fails. Phase B is never left running on the public internet.

## 1. Hard constraints (read before doing anything)

- **GPU: this container sees one MIG slice, `2g.20gb` (about 19.5 GB usable)**, not the 32 GB the owner was told.
  Check at the start with `nvidia-smi -L` and `python -c "import torch; print(torch.cuda.mem_get_info())"`. If it
  shows more, note it, but still design for 20 GB.
- **Host RAM: 8 GB cgroup cap.** The container resets if anonymous memory hits it. Run every model job under
  `interp/run_guarded.sh`. Never load model weights to `cpu`.
- **gpt-oss-20b in bf16 is 39 GB and does not fit.** The existing `gpt-oss-env` dequantizes MXFP4 to bf16 on load
  (triton 3.2 < 3.4), so `interp/gptoss.py` as written cannot run here any more. Do not try to make bf16 fit.
- **Other people's jobs:** CPDM training (`cpdm3d` env) may share the GPU. Never kill, pause or renice any process you
  did not start.
- **Do not modify** `interp/data/*`, `interp/cache/*` (except new directories you create), `results/*` (except
  `results/site/`), or the existing `gpt-oss-env`. Create a new conda env instead.
- **Numbers:** every number on the site must come from a file in `results/` that a script in this repo produced.
  Never type a metric by hand, never round in a flattering direction, never drop the text baseline. Placeholder
  numbers are forbidden. If a number does not exist yet, the site says "pending".
- **Labels stay out of the detector.** Files named `labels.jsonl` or `behavior.jsonl` are read only by training and
  evaluation code, never by the serving path.

## 2. Current results the site must show (source: `results/detector_demo/report.json`)

Leave-one-domain-out AUROC, target = "live instruction + harmful action in memory" vs the other three cells, 40
requests per set:

| | sound_2x2 (real case tools, 5 templates, no cue words) | scaleup_2x2x10 (one template per framing) |
|---|---|---|
| Probe, layers 14-19 | 0.84 | 0.90 |
| TF-IDF on the note text, no model | 0.87 | 0.88 |
| Probe, layer 1 | 0.81 | 0.81 |
| Prompt length only | 0.59 | 0.61 |

Honest status line for the site: *"The probe detects live planted instructions, but so far a bag-of-words model on
the memory text does about as well. The next experiment labels requests by whether the agent actually executes the
payload, to test whether activations carry information the text does not."* Read the numbers from the JSON at build
time and do not copy them from this table.

## 3. Phase A: static site (no GPU, do first)

### A1. Precompute examples
Script `site/build_static.py` (run with `~/.conda/envs/gpt-oss-env/bin/python`, CPU only):
- From `interp/data/sound_2x2` pick 8 requests: all four cells for 2 domains (Shopify and Slack).
- For each, get the **out-of-fold** probe score and the out-of-fold TF-IDF score from the same LODO procedure as
  `interp/detector_demo.py`. Import its functions; do not reimplement them. Add a small function to
  `detector_demo.py` that returns per-request out-of-fold scores if needed. Out-of-fold means the request's domain
  was never in that probe's training data.
- Emit `site/public/data/examples.json`: for each example, the user query, the memory note text, the cell (framing,
  payload), the probe score, the text-baseline score, and the decision threshold. The threshold is the LODO score
  at which the training folds had 10% FPR, computed in the script.
- Emit `site/public/data/results.json` from `results/detector_demo/report.json` (per-layer curves, baseline table,
  interaction test).

### A2. The page (`site/public/index.html`, one file plus `data/`)
Plain HTML/CSS/JS with no build step. One chart library from a CDN is fine (Chart.js). It must work at phone width.
Sections, in order:
1. **Hero.** Headline: "Catching sleeper instructions in an AI agent's memory, from the inside." Subline: one
   sentence on the threat (attackers plant instructions in agent memory that wait for a trigger, and output monitors
   see nothing until it fires). Two buttons: "See it work" (scrolls to 2) and "Read the results" (scrolls to 4).
2. **Interactive example.** On the left, a mock agent request: the system prompt collapsed, the user query, and the
   memory tool result with the planted note highlighted. On the right, two meters: "Activation probe" and "Text-only
   baseline", each with the threshold marked. Tabs or a dropdown switch between the 8 precomputed examples. Label it
   clearly: "Precomputed on held-out domains. Live sandbox available on request."
3. **How it works.** Three steps with a small diagram: request arrives → model reads it, and we read the residual
   stream at the end of the memory result (layers 14-19 of 24) → a linear probe outputs a score before the model
   writes a single token. Mention the provider-side threat model: the defender sees only the request and the
   model's internals.
4. **Results.** The per-layer chart (three lines plus the shaded 14-19 window) and the baseline bar chart, both from
   `results.json`. Under them, the honest status line from section 2, word for word.
5. **What's next.** Three bullets: behaviour-labelled evaluation (1,924 requests, held-out domains and templates,
   built and waiting for GPU time), a benign-payload control arm, and other models.
6. **Contact.** A mailto link and a GitHub link. **ASK** the owner for both. No form, no waitlist, no company name, no
   logo.

Tone: researcher, not startup. No "revolutionary", no "enterprise-grade", no "commitment detection" claims.

### A3. Hosting
Static files only, so GitHub Pages, Netlify or Vercel all work. **ASK** the owner which one, and whether the repo may be
public. Before publishing, check that `site/public/` contains no request ids from `labels.jsonl` beyond the 8
examples, no file paths from the host, and no email addresses except the contact one.

**Phase A done when:** the page opens from a local `python -m http.server`, all numbers trace to JSON files, it works
at 375 px wide, and the owner has approved it.

## 4. Phase B: live sandbox (GPU, 20 GB)

### B0. Gate: does gpt-oss-20b run in MXFP4 under HF transformers on this A100 slice? (half a day, max)
The checkpoint is about 13 GB in MXFP4. transformers keeps MXFP4 weights only with triton ≥ 3.4 and the `kernels`
package (`kernels-community/gpt-oss-triton-kernels` is already in `~/.cache/huggingface/hub`).
1. Create a new env: `conda create -n gptoss-mxfp4 python=3.11`, then install torch (a version whose bundled triton is
   ≥ 3.4), transformers 5.x, `kernels`, `fastapi`, `uvicorn` and `scikit-learn`. Record the exact versions in
   `site/ENV.md`.
2. Script `site/gate_mxfp4.py`, run under `interp/run_guarded.sh` with `HF_HUB_OFFLINE=1`:
   - Load with `device_map="cuda:0"` and **no** dequantize flag. Assert that the expert weights are still packed
     (not bf16). Print `torch.cuda.max_memory_allocated()`.
   - Forward one request from `interp/data/behavior_v1/requests.jsonl` with hidden states captured at
     `memory_result_end`. Record peak memory; it must stay under 17 GB.
   - Fidelity: teacher-force the 15 demo_v1 requests (`interp/data/demo_v1/outputs.jsonl`) and report argmax
     agreement with the recorded vLLM replies, the same metric as `interp/validate_and_capture.py`. vLLM also runs
     MXFP4, so expect ≥ 93%.
   - Check that `gptoss.check_edit` still passes with the hooks (port `hooks`, `ablate` and `check_edit` unchanged).
3. **If the gate passes,** continue to B1.
   **If MXFP4 kernels refuse sm80 (A100) or peak memory is over 17 GB,** stop and report to the owner with the error.
   The fallback is a per-layer dequantizing expert module: keep experts packed on the GPU and dequantize one layer's
   experts into a reused bf16 buffer during that layer's forward (about 1.5 GB transient). It is a real engineering
   task, so do not start it without the owner's go-ahead.

### B1. Re-capture and retrain (activations change between bf16 and MXFP4, so the old caches cannot train this probe)
- Port `interp/gptoss.py` to `site/model.py` for the new env: the same API, plus loading that keeps MXFP4. Capture only
  the `memory_result_end` position and layers 14-19 with a forward hook. Do not use `output_hidden_states` for all
  positions.
- Capture `interp/data/behavior_v1` into a **new** cache dir `interp/cache/behavior_v1_mxfp4/` (1,924 requests,
  resumable, under `run_guarded.sh`). Expect about 15-30 min.
- Train the serving probe on the pre-registered **train** split only (`split == "train"` in `labels.jsonl`). Target:
  `framing == "live"` (a live planted instruction in memory) vs inert or no note. Use the same probe recipe as
  `detector_demo.py` (standardise, then L2 logistic regression, C = 1.0, no tuning) on the layers-14-19 mean.
- Evaluate once on the **test** split. Report AUROC, TPR at 1% and 5% FPR, and the same metrics for a TF-IDF model on
  the memory text trained on the same split. Write `results/site/probe_eval.json`. Never touch `held_out_mixed`
  rows. If behaviour labels exist by then (`behavior.jsonl`), also report how well the probe predicts execution.
- Save `site/artifacts/probe.npz` (scaler mean and scale, coef, intercept, layers, position, threshold at 5% train
  FPR) and `site/artifacts/tfidf.pkl`.
- Add the Phase B numbers to the site as a separate block ("live model, MXFP4, held-out test split") next to the
  Phase A numbers. Do not merge or replace them.

### B2. Backend (`site/server.py`, FastAPI, one process, one model)
- On startup: load the model once and the probe artifacts, call `torch.cuda.set_per_process_memory_fraction` to stay
  under 18 GB, and run one warm-up forward.
- `GET /api/presets`: about 6 scenarios built from real `behavior_v1` cases (system prompt, tools, user query, memory
  entries).
- `POST /api/analyze` with `{preset_id, user_query, memory_notes: [str]}`:
  - Build the request with the same code as `demo/build_behavior_set.py:build_request` and render tokens with
    `demo/harmony_render.py`. Reject it if it is over 4,096 tokens. Each note may be at most 600 characters, with at
    most 5 notes.
  - Run one forward and score the probe at `memory_result_end`, plus the TF-IDF score. Return both scores, the
    threshold, and per-note attribution: the probe score with each note removed, one extra forward per note, at most
    5.
  - Optional `generate: true`: greedy generation until the first tool call or the end of the reply (cap 1,024 new
    tokens). Return the reply text and the parsed tool call, marked "not executed, sandbox". Use the case's success
    conditions to say whether the planted action would have fired.
- Concurrency: one `asyncio.Lock` around the GPU, with a queue depth of 3 and HTTP 429 beyond that. Timeout 120 s.
- Never log request contents to disk by default.
- Every model call in the server goes through `run_guarded.sh` limits: start the server itself under the guard.

### B3. Frontend changes
In section 2 of the page, add a "Live sandbox" tab that appears only when `/api/presets` answers. Pick a scenario,
edit or add memory notes, press Analyze, and see the two meters plus per-note attribution bars. Add a "Let the agent
respond" button for generation. Show the queue position while waiting.

### B4. Exposure (live demos only)
- Run the server on `127.0.0.1:8080`. For a scheduled call, expose it with a temporary tunnel (`cloudflared tunnel
  --url` or `ngrok`) behind a shared token header. Tear it down after the call. **ASK** the owner first: the cluster's
  acceptable-use policy may forbid public tunnels, and if it does, demo it over screen share instead.
- The static site's live tab points at the tunnel URL only when one is given (query parameter `?api=`), so the public
  site never depends on the GPU being up.

**Phase B done when:** the gate passes; `probe_eval.json` exists and is shown with its text baseline; a 5-note
request returns scores in under 5 s (without generation); peak GPU memory is logged under 18 GB; and the owner has
run one end-to-end demo.

## 5. Order of work and checkpoints with the owner

1. A1 → A2 → owner review → A3 publish. *(Shareable result; about a day.)*
2. B0 gate → report to the owner (pass or fail, peak memory, fidelity).
3. B1 → report `probe_eval.json` to the owner **before** it goes on the site.
4. B2 → B3 → owner runs a demo → B4 only for scheduled calls.

## 6. Definition of honest (applies to every page, post and email built from this)

- Always show the text baseline next to the probe.
- Say "detects a live planted instruction in memory", not "detects intent" or "detects commitment".
- State the sample sizes (40 requests per 2x2 set; the test split size for Phase B).
- The pilot's causal-intervention numbers (`results/full_reviewer_suite_report.json`, step 3) are invalid. Never cite them.
- The open question goes on the page, in plain words.
