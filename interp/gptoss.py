"""Interp helpers for gpt-oss-20b on the sleeper-detection request datasets (Hugging Face, bf16).

Run with the gpt-oss-env interpreter, under the memory guard:
    cd ~/sleeper_detection/interp && ./run_guarded.sh ~/.conda/envs/gpt-oss-env/bin/python <script>.py

Facts (checked 2026-09-30 on the A100-80GB): the checkpoint's MXFP4 weights are dequantized to bf16 on load
(triton 3.2 < 3.4), which takes ~75 s, 39 GiB of GPU memory and ~3.1 GiB of host memory. vLLM needs the GPU too,
so run one or the other. 24 layers, d_model 2880, 32 experts with top-4 routing, attention implementation "eager".

Main entry points
    load_dataset(dir)                      -> requests (defender view), labels, outputs keyed by id
    model()                                -> cached (model, tokenizer)
    forward(ids, positions=None, ...)      -> residual stream (+ router logits, attention) at chosen positions
    logit_lens(resid, top_k)               -> per-layer top tokens after the final norm + unembedding
    capture(dataset_dir, out_dir, ...)     -> cache activations for every request to disk, once
    load_cache(out_dir)                    -> stacked arrays + ids
    hooks({layer: fn})                     -> context manager for residual-stream edits (ablation, steering)
    teacher_force_ids(output)              -> prompt + the model's recorded vLLM reply, as token ids
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODEL_ID = "openai/gpt-oss-20b"
_MODEL = None
_TOK = None


# ----------------------------------------------------------------------------------------------- data
def load_dataset(path: str | Path) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    """requests (list, defender view), labels {id: ...} (evaluation only), outputs {id: ...}."""
    path = Path(path)
    read = lambda name: [json.loads(l) for l in open(path / f"{name}.jsonl")] if (path / f"{name}.jsonl").exists() else []  # noqa: E731
    return read("requests"), {r["id"]: r for r in read("labels")}, {r["id"]: r for r in read("outputs")}


def named_positions(record: dict, include_entries: bool = True) -> dict[str, int]:
    """Label-free token positions: last prompt token, user end, memory result end, and the last token of every
    stored-entry occurrence (entry<i>.occ<j>)."""
    pos = dict(record["positions"])
    if include_entries:
        for e in record["entry_spans"]:
            for j, (a, b) in enumerate(e["occurrences"]):
                pos[f"entry{e['entry_index']}.occ{j}.last"] = b - 1
    return pos


# ---------------------------------------------------------------------------------------------- model
def model(device: str = "cuda:0"):
    global _MODEL, _TOK
    if _MODEL is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device,
                                                      local_files_only=True)
        _MODEL.eval()
    return _MODEL, _TOK


@torch.no_grad()
def forward(ids: list[int], positions: Iterable[int] | None = None, router: bool = True, attentions: bool = False,
            logits_at: Iterable[int] | None = None) -> dict:
    """One forward pass. Returns (on CPU, float32):
         resid   [n_pos, n_layers+1, d]  residual stream after embeddings (index 0) and after each layer
         router  [n_pos, n_layers, n_experts]  router logits (if router=True)
         attn    list per layer of [heads, n_pos, seq] attention from the chosen positions (if attentions=True)
         logits  [n_logit_pos, vocab] final logits at logits_at (default: last position)
       positions default to every token."""
    m, _ = model()
    x = torch.tensor([ids], device=m.device)
    o = m(x, output_hidden_states=True, output_router_logits=router, output_attentions=attentions)
    sel = torch.arange(len(ids)) if positions is None else torch.tensor(list(positions))
    out = {"positions": sel.tolist(),
           "resid": torch.stack([h[0, sel] for h in o.hidden_states], dim=1).float().cpu()}
    if router and o.router_logits is not None:
        out["router"] = torch.stack([r[sel] for r in o.router_logits], dim=1).float().cpu()
    if attentions and o.attentions is not None:
        out["attn"] = [a[0][:, sel].float().cpu() for a in o.attentions]
    la = [len(ids) - 1] if logits_at is None else list(logits_at)
    out["logits"] = o.logits[0, la].float().cpu()
    return out


@torch.no_grad()
def logit_lens(resid: torch.Tensor, top_k: int = 5) -> list[list[tuple[str, float]]]:
    """resid [n_layers+1, d] for one position -> per layer, top_k (token, prob) after final norm + unembedding."""
    m, tok = model()
    h = resid.to(m.device, dtype=torch.bfloat16)
    probs = torch.softmax(m.lm_head(m.model.norm(h)).float(), dim=-1)
    p, i = probs.topk(top_k, dim=-1)
    return [[(tok.decode([int(t)]), float(q)) for t, q in zip(ii, pp)] for ii, pp in zip(i.cpu(), p.cpu())]


@contextlib.contextmanager
def hooks(edits: dict[int, Callable[[torch.Tensor], torch.Tensor]]):
    """Edit the residual stream after decoder layer L: edits = {L: fn(hidden [1, seq, d]) -> hidden}.
    Example (ablate a direction v at layer 12):  {12: lambda h: h - (h @ v)[..., None] * v}"""
    m, _ = model()
    handles = []
    for layer, fn in edits.items():
        def hook(_mod, _inp, output, fn=fn):
            if isinstance(output, tuple):
                return (fn(output[0]),) + output[1:]
            return fn(output)
        handles.append(m.model.layers[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def teacher_force_ids(output: dict) -> tuple[list[int], int]:
    """Prompt + the model's recorded vLLM reply as token ids, and the index where the reply starts.
    Built by build_dataset.py with the harmony encoder (outputs.jsonl: teacher_forced_ids, response_start)."""
    return output["teacher_forced_ids"], output["response_start"]


# ---------------------------------------------------------------------------------------------- cache
def capture(dataset_dir: str | Path, out_dir: str | Path, router: bool = True, include_entries: bool = True) -> Path:
    """Cache the residual stream (+ router logits) at every named position of every request, once.
    Writes out_dir/<id>.npz (resid float16 [n_pos, 25, 2880], router float16 [n_pos, 24, 32], logits_top) and
    out_dir/index.json with position names. Skips ids already cached (resumable)."""
    requests, _, _ = load_dataset(dataset_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for r in requests:
        if r["id"] in index and (out / f"{r['id']}.npz").exists():
            continue
        pos = named_positions(r, include_entries)
        names, idx = list(pos), list(pos.values())
        o = forward(r["input_ids"], positions=idx, router=router)
        top = torch.topk(torch.softmax(o["logits"][0], -1), 10)
        np.savez_compressed(out / f"{r['id']}.npz", resid=o["resid"].numpy().astype(np.float16),
                            router=o["router"].numpy().astype(np.float16) if router else np.zeros(0),
                            next_top_ids=top.indices.numpy(), next_top_p=top.values.numpy())
        index[r["id"]] = {"position_names": names, "positions": idx, "n_tokens": r["n_tokens"]}
        index_path.write_text(json.dumps(index, indent=1))
        print(f"cached {r['id']}  positions={len(idx)}", flush=True)
    return out


def load_cache(out_dir: str | Path, position: str = "last_prompt", ids: list[str] | None = None) -> dict:
    """Stack one named position across cached requests: resid [n, 25, d], router [n, 24, E], ids."""
    out = Path(out_dir)
    index = json.loads((out / "index.json").read_text())
    ids = ids or list(index)
    resid, router, kept = [], [], []
    for i in ids:
        names = index[i]["position_names"]
        if position not in names:
            continue
        k = names.index(position)
        z = np.load(out / f"{i}.npz")
        resid.append(z["resid"][k])
        if z["router"].size:
            router.append(z["router"][k])
        kept.append(i)
    return {"ids": kept, "resid": np.stack(resid), "router": np.stack(router) if router else None}
