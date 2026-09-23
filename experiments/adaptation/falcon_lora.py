"""Shared Falcon + LoRA plumbing for the adaptation experiment.

Loads the model exactly as satquery/specialists/falcon.py does (same flash_attn import patch,
same snapshot_download -> from_pretrained -> .to(device).eval() path), so what is trained here
is what the app runs.

Self-contained by design, like experiments/model_feasibility/: the app does not import this.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

MODEL_ID = "mehmetbayik/Falcon-Single-Instruction-Large"

# Measured 2026-09-21: 96 decoder attention Linears (self_attn + encoder_attn q/k/v/out_proj).
# Listed by discovery rather than a regex: peft 0.17 rejects a regex passed inside a list, and
# explicit names make the training config auditable.
DECODER_ATTN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "out_proj")


def make_without_flash_attn():
    """Falcon's remote code imports flash_attn behind an `if`; transformers' check still requires it.

    The original `get_imports` is captured before patching. Looking it up inside the body instead
    would resolve to the patched function itself and recurse until the stack runs out.
    """
    from transformers.dynamic_module_utils import get_imports as original

    def without_flash_attn(filename):
        return [imp for imp in original(filename) if imp != "flash_attn"]

    return without_flash_attn


def load_base(model_id: str = MODEL_ID, device: str = "cuda", dtype_name: str = "fp16"):
    """Return (model, processor, device, dtype), loaded the same way the app loads them."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoProcessor

    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    local_path = snapshot_download(model_id)
    with patch("transformers.dynamic_module_utils.get_imports", make_without_flash_attn()):
        model = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=True, torch_dtype=dtype)
        processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
    return model.to(device).eval(), processor, device, dtype


def decoder_attention_modules(model) -> list[str]:
    import torch

    return [name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and ".decoder." in name
            and name.rsplit(".", 1)[-1] in DECODER_ATTN_SUFFIXES]


def attach_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """Attach LoRA to the decoder attention projections. The vision tower stays frozen."""
    from peft import LoraConfig, get_peft_model

    targets = decoder_attention_modules(model)
    if not targets:
        raise RuntimeError("no decoder attention projections found: cannot attach LoRA")
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=targets))
    # LoRA params must be fp32 for a stable optimiser step even though the base stays fp16.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model, targets


def apply_adapter(model, adapter_path: str):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path).eval()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rgb(root: Path, row: dict) -> np.ndarray:
    with Image.open(Path(root) / row["image_path"]) as image:
        return np.asarray(image.convert("RGB"))


def normalise(answer: str) -> str:
    """Compare answers case- and punctuation-insensitively, so 'Yes.' matches 'yes'."""
    return "".join(c for c in str(answer).strip().lower() if c.isalnum() or c.isspace()).strip()
