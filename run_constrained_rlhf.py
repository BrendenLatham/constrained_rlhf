#!/usr/bin/env python3
"""Self‑contained driver for dual‑only constrained RLHF.
Works with ppo_utils.py  (compute_rewards included).
"""
import argparse, random, torch
from pathlib import Path
from datasets import load_dataset

from utils.modeling import load_unsloth_model
from extract_pairwise_tensors import build_hh_rlhf_tensors
from ppo_utils import (
    generate_batch,
    ppo_step,
    compute_rewards,
    RewardHead,
    _gather_logprobs,
)

def fit_head(delta, y, hidden, lr=2e-3, epochs=3):
    head = RewardHead(hidden).to(delta.device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    for _ in range(epochs):
        m = (2*y-1)*head(delta); loss = torch.nn.functional.softplus(-m).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="unsloth/Meta-Llama-3-8B-Instruct-bnb-4bit")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--flip_prob", type=float, default=0.3)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--lambda_cap", type=float, default=1.0)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_unsloth_model(args.model_name, device)
    ref_model = model.__class__.from_pretrained(args.model_name, device_map="auto", low_cpu_mem_usage=True).eval()
    hidden = model.config.hidden_size

    d1,y1 = build_hh_rlhf_tensors(model,tok,device,flip_prob=0.0)
    d2,y2 = build_hh_rlhf_tensors(model,tok,device,flip_prob=args.flip_prob)
    h1 = fit_head(d1,y1,hidden); h2 = fit_head(d2,y2,hidden)

    prompts = [r["prompt"] for r in load_dataset("anthropic/hh-rlhf","helpful",split="train")]
    lam = torch.tensor(0.0, device=device)

    for step in range(args.T):
        batch = generate_batch(model,tok,random.sample(prompts,args.batch_size),device=device)
        with torch.no_grad():
            ref_lp = _gather_logprobs(ref_model(batch["sequences"]).logits[:, :-1, :], batch["sequences"][:,1:])
        r1,r2 = compute_rewards(batch,model,h1,h2)
        adv = (r1+lam*r2).unsqueeze(1).expand_as(batch["logprobs"])
        stats = ppo_step(model,tok,batch,ref_lp,adv,kl_coef=args.eta)
        gap = r2.mean()-args.epsilon
        lam = torch.clamp(lam - args.alpha*(-gap),0.0,args.lambda_cap)
        if (step+1)%args.log_every==0:
            print(f"[{step+1:>5}] J={r1.mean():.3f} gap={gap:+.4f} λ={lam.item():.3f} KL={stats['kl']:.4f}")

if __name__=="__main__":
    main()
