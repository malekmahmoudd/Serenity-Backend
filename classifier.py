"""
classifier.py
=============
Loads the trained BiLSTM and its artifacts ONCE at import, and exposes
`score(text) -> {label: calibrated_probability}`.

FIVE ARTIFACTS SHIP TOGETHER. A model without its exact tokenizer, calibrators
and thresholds is not loadable, and mismatched versions fail silently rather
than loudly. All five live in backend/artifacts/:

    model_v5.keras             the network (Keras path)
    model_v5.onnx              same network, exported by export_onnx.py
                               (ONNX path -- optional, see below)
    tokenizer.json             JSON, not pickle -- pickles break across
                               TF versions and fail at service start-up
    calibrators.pkl            per-class isotonic regressors
    thresholds.json            per-class cut points, in CALIBRATED space
    label_map.json             explicit class index -> name

TWO SERVING PATHS
------------------
ONNX (preferred, when available): loads model_v5.onnx via onnxruntime and
tokenizes with lite_tokenizer.py -- neither imports TensorFlow. ~15MB runtime
instead of ~600MB, no graph tracing at import time. This is what
export_onnx.py exists to produce, and its own validation step is what makes
this path trustworthy: it will not promote model_v5.onnx unless its output
matched the Keras model within tight tolerance on real sentences.

Keras (fallback): used automatically if model_v5.onnx is missing, or if
USE_ONNX=false is set explicitly. Also what export_onnx.py itself loads to
produce the ONNX file in the first place -- this path is never removed, only
made optional at serve time.

Set USE_ONNX=false to force the Keras path even when the ONNX file exists
(useful for A/B-checking the two against real traffic before trusting ONNX
fully in production).

DEGRADED MODE
-------------
If the artifacts are missing entirely the module still imports and the server
still runs -- `available()` returns False and the decision layer falls back to
rule-only behaviour. That is deliberate: you should be able to boot the API and
exercise the whole request path before the model files are in place.
"""

from __future__ import annotations

import json
import os
import pickle
import threading

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
USE_ONNX = os.environ.get("USE_ONNX", "true").lower() != "false"

_state = {"ready": False, "error": None, "backend": None}
_lock = threading.Lock()

MODEL = None                # Keras model, only set on the Keras path
ONNX_SESSION = None          # onnxruntime session, only set on the ONNX path
ONNX_INPUT_NAME = None
TOKENIZER = None
CALIBRATORS = None
THRESHOLDS = None
ID2LABEL = None
LABELS = None
CONFIG = None
_pad_sequences = None


def _load_shared() -> tuple[bool, list]:
    """Label map, config, calibrators, thresholds -- identical on both paths."""
    global CALIBRATORS, THRESHOLDS, ID2LABEL, LABELS, CONFIG
    from text_cleaning import CLEANING_VERSION

    with open(os.path.join(ARTIFACT_DIR, "label_map.json")) as fh:
        lm = json.load(fh)
    ID2LABEL = {int(k): v for k, v in lm["id_to_label"].items()}
    LABELS = [ID2LABEL[i] for i in range(len(ID2LABEL))]

    # Train/serve parity check. A silent mismatch here degrades the model with
    # no error anywhere, which is the worst failure mode available.
    cfg_path = os.path.join(ARTIFACT_DIR, "preprocessing_config.json")
    CONFIG = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    trained_with = CONFIG.get("cleaning_version") or lm.get("cleaning_version")
    if trained_with and trained_with != CLEANING_VERSION:
        _state["error"] = (
            f"cleaning version mismatch: model trained with {trained_with}, "
            f"text_cleaning.py is {CLEANING_VERSION}. Retrain or restore "
            f"the matching cleaning code.")
        return False, []

    with open(os.path.join(ARTIFACT_DIR, "calibrators.pkl"), "rb") as fh:
        CALIBRATORS = pickle.load(fh)
    with open(os.path.join(ARTIFACT_DIR, "thresholds.json")) as fh:
        THRESHOLDS = json.load(fh)["thresholds"]
    return True, []


def _load_onnx() -> bool:
    global ONNX_SESSION, ONNX_INPUT_NAME, TOKENIZER, _pad_sequences
    import onnxruntime as ort
    from lite_tokenizer import load_tokenizer, pad_sequences

    TOKENIZER = load_tokenizer(os.path.join(ARTIFACT_DIR, "tokenizer.json"))
    _pad_sequences = pad_sequences
    ONNX_SESSION = ort.InferenceSession(
        os.path.join(ARTIFACT_DIR, "model_v5.onnx"),
        providers=["CPUExecutionProvider"])
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    return True


def _load_keras() -> bool:
    global MODEL, TOKENIZER, _pad_sequences
    # Keras 2 and Keras 3 disagree about where these live.
    try:
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
    except Exception:
        from keras.preprocessing.text import tokenizer_from_json
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except Exception:
        from keras.utils import pad_sequences
    from tensorflow import keras

    # MUST be imported before load_model. The decorator on
    # MaskedGlobalPooling1D only registers the class when the module is
    # actually executed -- without this, Keras finds the name in the saved
    # config and cannot construct it:
    #   Could not locate class 'MaskedGlobalPooling1D'
    from model_layers import CUSTOM_OBJECTS

    with open(os.path.join(ARTIFACT_DIR, "tokenizer.json")) as fh:
        TOKENIZER = tokenizer_from_json(fh.read())
    _pad_sequences = pad_sequences

    MODEL = keras.models.load_model(
        os.path.join(ARTIFACT_DIR, "model_v5.keras"),
        custom_objects=CUSTOM_OBJECTS, compile=False)
    return True


def _load() -> None:
    """Load everything. Called once at import; safe to call again."""
    onnx_path = os.path.join(ARTIFACT_DIR, "model_v5.onnx")
    keras_path = os.path.join(ARTIFACT_DIR, "model_v5.keras")
    use_onnx = USE_ONNX and os.path.exists(onnx_path)

    shared_required = ["calibrators.pkl", "thresholds.json", "label_map.json"]
    missing = [f for f in shared_required
               if not os.path.exists(os.path.join(ARTIFACT_DIR, f))]
    if missing or not (os.path.exists(onnx_path) or os.path.exists(keras_path)):
        if missing:
            _state["error"] = f"missing artifacts in {ARTIFACT_DIR}: {missing}"
        else:
            _state["error"] = (
                f"neither model_v5.onnx nor model_v5.keras found in "
                f"{ARTIFACT_DIR}. Copy them from the Kaggle notebook output, "
                f"or run export_onnx.py after copying the .keras file.")
        return

    try:
        ok, _ = _load_shared()
        if not ok:
            return   # _load_shared already set _state["error"]

        if use_onnx:
            _load_onnx()
            _state["backend"] = "onnx"
        else:
            if USE_ONNX and not os.path.exists(onnx_path):
                print(f"[classifier] USE_ONNX is true but {onnx_path} does not "
                      f"exist -- falling back to Keras. Run export_onnx.py to "
                      f"enable the lighter path.", flush=True)
            _load_keras()
            _state["backend"] = "keras"

        # Warm up. First predict() builds the graph (Keras) or the session
        # cache (ONNX) and takes noticeably longer than steady state; do it
        # now rather than on a user's first message.
        score("warming up the model with a sentence of ordinary length")

        _state["ready"] = True
        _state["error"] = None
    except Exception as exc:  # noqa: BLE001
        if use_onnx:
            # ONNX path failed for some reason (corrupt file, version
            # mismatch in onnxruntime) -- try Keras once before giving up
            # entirely, since it is the artifact more likely to still work.
            print(f"[classifier] ONNX path failed ({type(exc).__name__}: {exc}), "
                  f"trying Keras fallback...", flush=True)
            try:
                _load_keras()
                _state["backend"] = "keras (onnx fallback)"
                score("warming up the model with a sentence of ordinary length")
                _state["ready"] = True
                _state["error"] = None
                return
            except Exception as exc2:  # noqa: BLE001
                _state["error"] = (f"ONNX failed ({exc}); Keras fallback also "
                                   f"failed: {type(exc2).__name__}: {exc2}")
                return
        _state["error"] = f"{type(exc).__name__}: {exc}"


def available() -> bool:
    return _state["ready"]


def status() -> dict:
    return {"ready": _state["ready"], "error": _state["error"],
            "backend": _state["backend"], "labels": LABELS,
            "artifact_dir": ARTIFACT_DIR, "thresholds": THRESHOLDS}


def score(text: str) -> dict:
    """
    Clean -> tokenize -> predict -> calibrate.

    Returns {label: calibrated_probability} for every class. Calibrated, so a
    0.7 means roughly 70% -- that is what makes the numbers safe to put in an
    LLM prompt. Identical output shape and semantics on both backends; the
    caller never needs to know which one is serving.
    """
    import numpy as np
    from text_cleaning import clean_text

    cleaned = clean_text(text)
    seq = TOKENIZER.texts_to_sequences([cleaned])
    maxlen = int(CONFIG.get("maxlen", 256)) if CONFIG else 256
    X = _pad_sequences(seq, maxlen=maxlen, padding="post",
                       truncating="post").astype("int32")
    if X.sum() == 0:
        # every token unknown -> mark as a single OOV so the sequence is not
        # all-padding, which the pooling layer cannot represent. Both Keras's
        # Tokenizer and LiteTokenizer expose .oov_token and .word_index the
        # same way, so this needs no backend-specific branch.
        oov_token = getattr(TOKENIZER, "oov_token", None) or "<OOV>"
        oov = TOKENIZER.word_index.get(oov_token, 1)
        X[0, 0] = oov

    if _state["backend"] and _state["backend"].startswith("onnx"):
        raw = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: X})[0][0]
    else:
        raw = MODEL.predict(X, verbose=0)[0]

    # The saved model already ends in softmax/sigmoid; if it emits logits,
    # squash them. Detect by range rather than by configuration.
    if raw.min() < 0.0 or raw.max() > 1.0:
        raw = 1.0 / (1.0 + np.exp(-raw))

    out = {}
    for i, name in enumerate(LABELS):
        cal = CALIBRATORS.get(name)
        v = float(cal.predict([float(raw[i])])[0]) if cal is not None else float(raw[i])
        out[name] = round(v, 4)
    return out


def fired(scores: dict) -> list[str]:
    """Labels whose calibrated score clears their own tuned threshold."""
    if not THRESHOLDS:
        return []
    hits = [(k, v) for k, v in scores.items() if v >= THRESHOLDS.get(k, 0.5)]
    hits.sort(key=lambda kv: -kv[1])
    return [k for k, _ in hits]


with _lock:
    _load()

