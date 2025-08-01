"""
utils/modeling.py
-----------------
Minimal helper so other code can call

    from utils.modeling import load_unsloth_model
"""
from unsloth import FastLanguageModel


def load_unsloth_model(model_name: str, device: str = "cuda"):
    """
    Returns (model, tokenizer) already loaded in 4-bit and mapped to GPU.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model, tokenizer
