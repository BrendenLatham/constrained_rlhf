#!/usr/bin/env python3
"""run_constrained_rlhf.py

End‑to‑end driver for **dual‑only constrained RLHF** on Llama‑3‑8B‑4bit.
It combines the new helper modules you just added:
  • `extract_pairwise_tensors.py` – builds Oracle‑1 / Oracle‑2 datasets.
  • `ppo_utils.py`               – prompt generation + PPO update.

Pipeline
--------
1.  Load quantised model (UnsLoTH).
2.  Build pairwise tensors from Anthropic HH‑RLHF (helpful split).
    – Oracle‑1:   ground‑truth labels.
    – Oracle‑2:   random label flips with probability `flip_prob`.
3.  Fit *two* reward heads by offline MLE (frozen LM).
4.  Sample prompts and run PPO updates while a dual variable `lambda_`
    is updated via projected gradient to enforce the constraint
            E_pi[r2]  >=  epsilon.
5.  Save the policy, reward heads, and the final λ.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

# ────────────────────────────────────────────────────────────────────────────
# Local helper modules (added earlier)
# ────────────────────────────────────────────────────────────────────────────
from extract_pairwise_tensors import build_hh_rlhf_tensors
from ppo_utils import (
    PromptDataset,
    generate_batch,
    ppo_step,
    compute_rewards,
    _gather_logprobs,  # helper reused for π₀ log‑probs
)

# ────────────────────────────────────────────────────────────────────────────
# UnsLoTH loader (same util as in your original repo)
# ────────────────────────────────────────────────────────────────────────────
try:
    from utils.modeling import load_unsloth_model
except ImportError as err:
    raise RuntimeError("Cannot import UnsLoTH loader – add original repo to PYTHONPATH") from err

# ────────────────────────────────────────────────────────────────────────────
# Reward‑head definition & offline MLE
# ────────────────────────────────────────────────────────────────────────────
class RewardHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, hidden):  # hidden: (..., hidden_dim)
        return self.proj(hidden).squeeze(-1)  # (...,)


def fit_reward_head(delta: torch.Tensor, label: torch.Tensor,
                     hidden_dim: int, lr: float = 2e-3, epochs: int = 3) -> RewardHead:
    """Offline logistic‑loss MLE on (delta, label)."""
    device = delta.device
    head = RewardHead(hidden_dim).to(device)
    opt = optim.AdamW(head.parameters(), lr=lr)

    for _ in range(epochs):
        logits = head(delta)
        margins = (2 * label - 1) * logits  # {0,1}→{‑1,+1}
        loss = torch.nn.functional.softplus(-margins).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()

# ────────────────────────────────────────────────────────────────────────────
# Dual‑training loop
# ────────────────────────────────────────────────────────────────────────────

def dual_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load base & reference models (UnsLoTH quantised)
    print("► Loading base model …")
    model, tokenizer = load_unsloth_model(args.model_name, device)
    hidden_dim = model.config.hidden_size

    ref_model = model.__class__.from_pretrained(
        args.model_name, low_cpu_mem_usage=True, device_map="auto"
    ).eval()

    # 2. Build pairwise tensors for both oracles
    print("► Building pairwise tensors … (this may embed responses on first run)")
    delta1, y1 = build_hh_rlhf_tensors(model, tokenizer, device,
                                       flip_prob=0.0)
    delta2, y2 = build_hh_rlhf_tensors(model, tokenizer, device,
                                       flip_prob=args.flip_prob)

    # 3. Fit reward heads (offline MLE)
    print("► Fitting reward heads …")
    head1 = fit_reward_head(delta1, y1, hidden_dim, args.mle_lr, args.mle_epochs)
    head2 = fit_reward_head(delta2, y2, hidden_dim, args.mle_lr, args.mle_epochs)

    # 4. Prompt pool (HH‑RLHF prompts)
    hh_ds = load_dataset("anthropic/hh-rlhf", "helpful", split="train")
    prompt_pool: List[str] = [rec["prompt"] for rec in hh_ds]

    # 5. Dual variable & step size
    lambda_ = torch.tensor(0.0, device=device)
    alpha = args.alpha  # fixed step size; could be adaptive

    # 6. Training loop
    print("► Starting dual‑PPO loop …")
    for step in range(args.T):
        # ‑‑ sample prompts & generate responses
        prompts = random.sample(prompt_pool, args.batch_size)
        batch = generate_batch(model, tokenizer, prompts,
                               max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature,
                               top_p=args.top_p,
                               device=device)

        # π₀ log‑probs for KL term
        with torch.no_grad():
            ref_logits = ref_model(batch["sequences"]).logits
        ref_logprobs = _gather_logprobs(ref_logits[:, :-1, :],
                                        batch["sequences"][:, 1:])

        # rewards & advantage
        r1, r2 = compute_rewards(batch, model, head1, head2, tokenizer)
        adv = (r1 + lambda_ * r2).unsqueeze(1).expand_as(batch["logprobs"])

        # PPO update
        stats = ppo_step(model, tokenizer, batch, ref_logprobs, adv,
                         clip_ratio=0.2, kl_coef=args.eta, lr=args.lr)

        # dual PGD step
        gap = r2.mean() - args.epsilon  # positive ⇒ over‑satisfy constraint
        lambda_ = torch.clamp(lambda_ - alpha * (-gap), 0.0, args.lambda_cap)

        if (step + 1) % args.log_every == 0:
            print(
                f"[{step+1:>6d}] J≈{r1.mean():6.3f}  gap={gap.item():+7.4f}  "
                f"λ={lambda_.item():5.3f}  KL={stats['kl']:.4f}  loss={stats['loss']:.4f}"
            )

    # 7. Save artefacts
    print("► Saving artefacts …")
    out_dir = Path(args.output_dir);
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "policy")
    tokenizer.save_pretrained(out_dir / "policy")
    head1.cpu().save_pretrained(out_dir / "reward1_head")
    head2.cpu().save_pretrained(out_dir / "reward2_head")
    torch.save(lambda_.cpu(), out_dir / "lambda.pt")


# ────────────────────────────────────────────────────────────────────────────
# Argument parser & entry point
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="unsloth/llama-3-8b-4bit")

    # dataset / oracle2 noise
    ap.add_argument("--flip_prob", type=float, default=0.3,
                    help="Random‑flip probability for Oracle‑2 labels")

    # dual parameters
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--lambda_cap", type=float, default=1.0)

    # PPO parameters
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    # offline MLE
    ap.add_argument("--mle_lr", type=float, default=2e-3)
    ap.add_argument("--mle_epochs", type=int, default=3)

    # training loop
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)

    # misc
    ap.add_argument("--output_dir", default="outputs/constrained_rlhf")

    args = ap.parse_args()

    dual_loop(args)
