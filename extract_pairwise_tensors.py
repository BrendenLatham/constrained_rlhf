# extract_pairwise_tensors.py
"""Utilities for building pair‑wise preference tensors from the
Anthropic HH‑RLHF *helpful* split, with optional random label flips to
create a second (noisy) oracle.

The main public function is

    build_hh_rlhf_tensors(model, tokenizer, device,
                          cache_dir=".cache/embeds",
                          flip_prob=0.0)

It returns two tensors
    delta  – (N, hidden_dim)  where delta_i = h(chosen) – h(rejected)
    labels – (N,)             0/1  (after optional flipping)

*Caching.*  The first call stores a `pt` file containing a dictionary
{text -> hidden_vector} under the given `cache_dir`.  Subsequent calls
reuse embeddings, so the reward‑head MLE costs only ~2 s even for the
full 160 k pairs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

__all__ = ["build_hh_rlhf_tensors"]


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _load_cache(path: Path):
    if path.is_file():
        return torch.load(path)
    return {}


def _save_cache(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------

def build_hh_rlhf_tensors(
    model,
    tokenizer,
    device: torch.device,
    *,
    cache_dir: str | Path = ".cache/embeds",
    flip_prob: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (delta, label) tensors ready for MLE.

    * `flip_prob` = 0.0 → Oracle‑1 (ground‑truth helpful preferences)
      `flip_prob` > 0   → Oracle‑2 (randomly‑flipped labels)
    """
    assert 0.0 <= flip_prob < 1.0, "flip_prob must be in [0,1)."

    ds = load_dataset("anthropic/hh-rlhf", "helpful", split="train")
    N = len(ds)

    cache_dir = Path(cache_dir)
    embed_cache_file = cache_dir / "hh_rlhf_helpful_embeds.pt"
    embed_cache = _load_cache(embed_cache_file)  # dict[str, torch.FloatTensor]

    # We'll use mean pooled last hidden layer as ϕ(text).
    hidden_dim = model.config.hidden_size
    embeds = embed_cache  # alias

    def _encode(text: str) -> torch.Tensor:
        if text in embeds:
            return embeds[text]
        tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.no_grad():
            out = model.model(**tok, output_hidden_states=True, return_dict=True)
            h = out.hidden_states[-1]  # (1, seq, hidden)
            vec = h.mean(dim=1).squeeze(0).cpu()  # (hidden,)
        embeds[text] = vec
        return vec

    delta_list = []
    label_list = []

    gen = tqdm(ds, desc="Building preference tensors")
    for ex in gen:
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        h_c = _encode(chosen)
        h_r = _encode(rejected)

        delta = h_c - h_r  # preferred minus non‑preferred
        label = 1  # chosen preferred by oracle‑1
        if flip_prob > 0.0 and torch.rand(()) < flip_prob:
            label = 1 - label  # flip 0↔1
            delta = -delta     # keep convention: label=1 ⇒ first arg preferred

        delta_list.append(delta)
        label_list.append(label)

    # Save cache if new embeddings were added
    if len(embeds) > len(embed_cache):
        _save_cache(embeds, embed_cache_file)

    delta_tensor = torch.stack(delta_list).to(device)
    label_tensor = torch.tensor(label_list, dtype=torch.float32, device=device)
    return delta_tensor, label_tensor
