import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "kor_Hang"
TGT_LANG = "eng_Latn"
MAX_NEW_TOKENS = 128

_lock = threading.Lock()
_tokenizer = None
_model = None
_device = None
_tgt_token_id = None


def _load():
    global _tokenizer, _model, _device, _tgt_token_id
    if _model is not None:
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(_device)
    _model.eval()
    _tgt_token_id = _tokenizer.convert_tokens_to_ids(TGT_LANG)


def translate(text):
    text = text.strip()
    if not text:
        return ""
    with _lock:
        _load()
        inputs = _tokenizer(text, return_tensors="pt", truncation=True).to(_device)
        with torch.no_grad():
            out = _model.generate(
                **inputs,
                forced_bos_token_id=_tgt_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=1,
            )
        return _tokenizer.batch_decode(out, skip_special_tokens=True)[0]
