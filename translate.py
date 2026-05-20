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


def _vram(tag):
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            print(f"[translate] {tag} vram free={free/1e9:.2f}GB total={total/1e9:.2f}GB",
                  flush=True)
        except Exception as e:
            print(f"[translate] {tag} vram query failed: {e}", flush=True)


def _load():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    print("[translate] _load() start", flush=True)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[translate] device={_device}", flush=True)
    _vram("pre-load")

    print(f"[translate] loading tokenizer {MODEL_NAME} ...", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("[translate] tokenizer ready", flush=True)

    print("[translate] building BitsAndBytesConfig (4-bit nf4) ...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("[translate] bnb config ready", flush=True)

    print(f"[translate] loading model {MODEL_NAME} (this may take a while) ...",
          flush=True)
    try:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb,
            device_map="auto",
            use_safetensors=True,
        )
    except Exception as e:
        print(f"[translate] from_pretrained FAILED: {type(e).__name__}: {e}",
              flush=True)
        raise
    print("[translate] model loaded + quantized on device", flush=True)
    _vram("post-load")

    _model.eval()
    print("[translate] model.eval() done; _load() complete", flush=True)


def translate(text):
    text = text.strip()
    if not text:
        return ""
    print(f"[translate] translate() called text={text!r}", flush=True)
    with _lock:
        _load()
        print("[translate] applying chat template ...", flush=True)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        inputs = _tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(_device)
        print(f"[translate] inputs ready shape={tuple(inputs.shape)} -> generate()", flush=True)
        with torch.no_grad():
            out = _model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=_tokenizer.eos_token_id,
            )
        print(f"[translate] generate() done out_shape={tuple(out.shape)}", flush=True)
        gen = out[0][inputs.shape[1]:]
        result = _tokenizer.decode(gen, skip_special_tokens=True).strip()
        print(f"[translate] decoded result={result!r}", flush=True)
        return result
