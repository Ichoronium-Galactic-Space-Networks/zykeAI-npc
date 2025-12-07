from functools import lru_cache
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _quant_config(quantization: Optional[str]) -> Optional[BitsAndBytesConfig]:
    if quantization is None:
        return None
    quantization = quantization.lower()
    if quantization not in {"4bit", "8bit"}:
        raise ValueError("quantization must be one of: None, '4bit', '8bit'")
    if quantization == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    return BitsAndBytesConfig(load_in_8bit=True)


@lru_cache(maxsize=4)
def load_model_with_adapter(
    base_model: str,
    tokenizer_path: str,
    adapter_path: Optional[str] = None,
    quantization: Optional[str] = None,
):
    """
    Load a GPT-2 model and tokenizer, optionally applying a PEFT adapter and quantization.
    Results are cached by (base_model, adapter_path, quantization, tokenizer_path).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = _quant_config(quantization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16 if quant_cfg else None,
        device_map="auto" if quant_cfg else None,
    )

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required to load adapters") from exc
        model = PeftModel.from_pretrained(model, adapter_path)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if not quant_cfg:
        model.to(device)
    model.eval()
    return model, tokenizer, device
