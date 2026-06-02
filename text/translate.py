import os
import threading

# Project root is one level up from this file (text/translate.py → project/).
# Keep the cache at <project>/.hf_cache so it survives package moves.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, ".hf_cache"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Current: 7B + bitsandbytes 4-bit. Slower (~3 tok/s) but uses few-shot examples
# as guidance rather than copy-paste menu (3B failure mode).
# Faster swap once autoawq installs: change MODEL_NAME to "Qwen/Qwen2.5-7B-Instruct-AWQ"
# and switch _load() to the AWQ path (no BitsAndBytesConfig, just torch_dtype=float16).
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 96

SYSTEM_PROMPT = (
    "You translate Korean to English. The source is wuxia/xianxia webnovel or manhwa.\n"
    "RULES:\n"
    "1. Output ONLY the English translation of the given input. No quotes, no explanation, no romanization.\n"
    "2. Translate ONLY what is in the input. Do NOT add words, names, or context that are not in the source.\n"
    "3. If the input is short or fragmentary, give a literal short translation. Do not invent dialogue or guess what the speaker meant.\n"
    "4. For Korean martial-sect names rendered with Chinese hanja meanings, prefer the standard wuxia rendering (e.g. mountain+sect compounds = 'X Sect').\n"
    "5. Honorifics: keep when they read naturally (Master, Senior Brother, etc.).\n"
    "\n"
    "EXAMPLES:\n"
    "Korean: 망했다고?\n"
    "English: It fell?\n"
    "\n"
    "Korean: 화산파\n"
    "English: Mount Hua Sect\n"
    "\n"
    "Korean: 사부님!\n"
    "English: Master!\n"
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


def warmup(text="안녕"):
    """Force model load + one generation pass to pay CUDA kernel JIT cost upfront."""
    print("[translate] warmup() begin", flush=True)
    translate(text)
    print("[translate] warmup() complete", flush=True)


def translate(text):
    import time
    text = text.strip()
    if not text:
        return ""
    with _lock:
        _load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        t0 = time.perf_counter()
        enc = _tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = enc["input_ids"].to(_device)
        attention_mask = enc["attention_mask"].to(_device)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = _model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=_tokenizer.eos_token_id,
            )
        torch.cuda.synchronize() if _device == "cuda" else None
        t2 = time.perf_counter()
        gen = out[0][input_ids.shape[1]:]
        result = _tokenizer.decode(gen, skip_special_tokens=True).strip()
        n_in = input_ids.shape[1]
        n_out = gen.shape[0]
        print(f"[translate] in={n_in}tok out={n_out}tok "
              f"tok={t1-t0:.3f}s gen={t2-t1:.3f}s "
              f"({n_out/(t2-t1):.1f} tok/s)\n"
              f"           KO: {text!r}\n"
              f"           EN: {result!r}", flush=True)
        return result
