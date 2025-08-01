# ppo_utils.py – clean, no self‑import
import torch, random
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

__all__ = [
    "PromptDataset",
    "generate_batch",
    "ppo_step",
    "RewardHead",
    "_gather_logprobs",
]


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx]


def _gather_logprobs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(2, ids.unsqueeze(-1)).squeeze(-1)


def generate_batch(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompts, *,
                   max_new_tokens=256, temperature=1.0, top_p=0.9, device=None):
    device = device or model.device
    toks = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        gen = model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    seq = gen.sequences
    prompt_len = toks.input_ids.size(1)
    with torch.no_grad():
        logits = model(seq).logits
    lp = _gather_logprobs(logits[:, :-1, :], seq[:, 1:])
    mask = torch.zeros_like(lp, dtype=torch.bool)
    mask[:, : prompt_len - 1] = True
    return {
        "prompts": prompts,
        "sequences": seq,
        "prompt_len": prompt_len,
        "logprobs": lp,
        "prompt_mask": mask,
    }


def ppo_step(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch: dict,
             reference_logprobs: torch.Tensor, advantage: torch.Tensor, *,
             clip_ratio=0.2, kl_coef=0.01, lr=1e-5):
    opt = getattr(model, "_ppo_opt", None)
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model._ppo_opt = opt
    seq = batch["sequences"]
    logits_new = model(seq).logits
    lp_new = _gather_logprobs(logits_new[:, :-1, :], seq[:, 1:])
    lp_old = batch["logprobs"].detach()
    mask = ~batch["prompt_mask"]

    ratio = torch.exp((lp_new - lp_old) * mask)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
    ppo_loss = -torch.sum(torch.minimum(unclipped, clipped) * mask) / mask.sum()

    kl = (lp_new - reference_logprobs) * mask
    loss = ppo_loss + kl_coef * kl.mean()

    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return {"loss": loss.item(), "kl": kl.mean().item()}


class RewardHead(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = torch.nn.Linear(hidden, 1)
    def forward(self, h):
        return self.proj(h).squeeze(-1)
