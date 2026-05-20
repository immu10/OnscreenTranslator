import os
import threading

os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache"),
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "You are a translator. Translate Korean text to natural English.\n"
    "Context: the source is a wuxia/xianxia webnovel or manhwa, with martial sects, cultivators, and honorifics.\n"
    "Rules:\n"
    "- Output ONLY the English translation. No explanations, no quotes, no romanization.\n"
    "- Preserve proper nouns naturally (e.g., 화산파 -> Mount Hua Sect, not 'volcano sect').\n"
    "- Keep honorifics where they read naturally in English (Master, Senior Brother, etc.).\n"
    "- If the input is a single short label (UI text), translate concisely."
)

_lock = threading.Lock()
_tokenizer = None
_model = None
_device = None


def _load():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        use_safetensors=True,
    )
    _model.eval()


def translate(text):
    text = text.strip()
    if not text:
        return ""
    with _lock:
        _load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        inputs = _tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(_device)
        with torch.no_grad():
            out = _model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=_tokenizer.eos_token_id,
            )
        gen = out[0][inputs.shape[1]:]
        return _tokenizer.decode(gen, skip_special_tokens=True).strip()
