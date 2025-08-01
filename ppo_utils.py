# ppo_utils.py
"""Utilities for prompt sampling, response generation, and PPO updates.

This is a *minimal* implementation that stays close to the dual-only
theory: we optimise the KL‑regularised objective with a single PPO step
per batch and include the additional advantage term `lambda_ * r2`.

It relies only on PyTorch (no TRL) so you can read every line.  Designed
for 4‑bit or 8‑bit quantised Llama‑3 on an A100‑40GB.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Prompt dataset
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    """Simple wrapper over a list of prompt strings."""
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# ---------------------------------------------------------------------------
# Helper: log‑probs of generated tokens
# ---------------------------------------------------------------------------

def _gather_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return token‑level log P for `input_ids` under `logits`."""
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)


# ---------------------------------------------------------------------------
# Generation + batch preparation
# ---------------------------------------------------------------------------

def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device | None = None,
):
    """Generate responses and return tensors needed for PPO."""
    device = device or model.device

    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    B, prompt_len = prompt_tokens.input_ids.shape

    with torch.no_grad():
        gen_out = model.generate(
            **prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = gen_out.sequences            # (B, prompt_len + new_len)
    new_tokens = sequences[:, prompt_len:]   # (B, new_len)
    full_len = sequences.size(1)

    # Compute logprobs under *current* policy
    with torch.no_grad():
        logits = model(sequences).logits  # (B, full_len, vocab)
    logprobs = _gather_logprobs(logits[:, :-1, :], sequences[:, 1:])  # align

    # Mask out prompt tokens when computing advantages / losses
    prompt_mask = torch.zeros_like(logprobs, dtype=torch.bool)
    prompt_mask[:, : prompt_len - 1] = True  # minus 1 because of shift

    # Return everything as a dict
    batch = {
        "prompts": prompts,
        "sequences": sequences,
        "prompt_len": prompt_len,
        "logprobs": logprobs,  # (B, T‑1)
        "prompt_mask": prompt_mask,
    }
    return batch


# ---------------------------------------------------------------------------
# PPO step (single minibatch, clipped, no entropy bonus for simplicity)
# ---------------------------------------------------------------------------

def ppo_step(
    model,
    tokenizer,
    batch,
    reference_logprobs: torch.Tensor,
    advantage: torch.Tensor,
    *,
    clip_ratio: float = 0.2,
    kl_coef: float = 0.01,
    lr: float = 1e-5,
):
    """Run one PPO update on the generated batch.

    Parameters
    ----------
    reference_logprobs : token‑level log P under π₀, same shape as batch["logprobs"].
    advantage          : token‑level advantage signal (broadcastable).
    """
    optimiser = getattr(model, "_ppo_optimiser", None)
    if optimiser is None:
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
        model._ppo_optimiser = optimiser  # cache

    # Recompute logprobs with gradients
    sequences = batch["sequences"]
    logits = model(sequences).logits
    logprobs_new = _gather_logprobs(logits[:, :-1, :], sequences[:, 1:])

    logprobs_old = batch["logprobs"].detach()

    # Only learning on response tokens
    mask = ~batch["prompt_mask"]

    ratio = torch.exp((logprobs_new - logprobs_old) * mask)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    ppo_loss = -torch.sum(torch.minimum(unclipped, clipped) * mask) / mask.sum()

    # KL penalty toward reference
    kl = (logprobs_new - reference_logprobs) * mask
    kl_loss = kl_coef * kl.mean()

    loss = ppo_loss + kl_loss

    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()

    return {
        "loss": loss.item(),
        "ppo_loss": ppo_loss.item(),
        "kl": kl.mean().item(),
    }


# ---------------------------------------------------------------------------
# Reward computation helper
# ---------------------------------------------------------------------------

def compute_rewards(batch, model, head1, head2, tokenizer):
    """Return sequence‑level rewards r1, r2 (B,) by running reward heads on the
    *response* embeddings (mean pooled).
    """
    sequences = batch["sequences"]
    prompt_len = batch["prompt_len"]

    with torch.no_grad():
        outputs = model(sequences, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]      # (B, T, H)
        resp_emb = hidden[:, prompt_len - 1 :].mean(dim=1)  # mean pool response

    r1 = head1(resp_emb)
    r2 = head2(resp_emb)
    return r1, r2
