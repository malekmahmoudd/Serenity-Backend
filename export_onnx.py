"""
export_onnx.py
==============
Run ONCE, offline, whenever model_v5.keras changes. Not part of the request
path -- this is a build step, same category as training itself.

    venv\\Scripts\\python.exe export_onnx.py

WHAT IT DOES
--------------
1. Loads the trained Keras model (needs model_layers.CUSTOM_OBJECTS, same as
   classifier.py's Keras path).
2. Converts it to ONNX via tf2onnx.
3. Runs BOTH the Keras model and the ONNX Runtime session on the same batch
   of real, varied inputs, and asserts the outputs match within tolerance.
   An export that "succeeds" but produces different numbers is worse than one
   that fails loudly -- classifier.py switches serving traffic onto this file
   without a human in the loop, so nothing should trust it un-verified.
4. Writes artifacts/model_v5.onnx only if validation passes.

WHY THIS MATTERS: 74.9MB Keras model + TensorFlow (~600MB import, several
seconds of cold-start graph building) becomes an ~19MB ONNX file + onnxruntime
(~15MB import, no graph tracing at request time). On a platform that spins
containers up and down (Render/Railway free tier, serverless), that's the
difference between a slow first request and a slow first WEEK.
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
KERAS_PATH = os.path.join(ARTIFACT_DIR, "model_v5.keras")
ONNX_PATH = os.path.join(ARTIFACT_DIR, "model_v5.onnx")

# Same tolerance test_persistence.py-style files use elsewhere in this repo:
# tight enough to catch a real conversion bug, loose enough to survive
# ordinary float32 rounding differences between two different runtimes.
ATOL = 1e-4
RTOL = 1e-3


def main() -> None:
    t0 = time.time()
    sys.path.insert(0, HERE)

    if not os.path.exists(KERAS_PATH):
        sys.exit(f"FATAL: {KERAS_PATH} not found. Nothing to export.")

    print("loading Keras model...")
    from model_layers import CUSTOM_OBJECTS
    from tensorflow import keras
    import numpy as np
    import tensorflow as tf

    model = keras.models.load_model(KERAS_PATH, custom_objects=CUSTOM_OBJECTS,
                                    compile=False)
    print(f"  loaded in {time.time()-t0:.1f}s. inputs: {model.input_shape}, "
          f"outputs: {model.output_shape}")

    maxlen = model.input_shape[1]
    if maxlen is None:
        # some export configs report a dynamic axis; fall back to the
        # preprocessing config the rest of the app already trusts
        cfg = json.load(open(os.path.join(ARTIFACT_DIR, "preprocessing_config.json")))
        maxlen = int(cfg["maxlen"])
    print(f"  sequence length: {maxlen}")

    print("\nconverting to ONNX (this can take a minute)...")
    import tf2onnx
    spec = (tf.TensorSpec((None, maxlen), tf.int32, name="tokens"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec,
                                               opset=15)

    tmp_path = ONNX_PATH + ".tmp"
    with open(tmp_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())
    onnx_size = os.path.getsize(tmp_path) / 1e6
    keras_size = os.path.getsize(KERAS_PATH) / 1e6
    print(f"  wrote {tmp_path}  ({onnx_size:.1f} MB, vs {keras_size:.1f} MB "
          f"Keras -- {keras_size/onnx_size:.1f}x smaller)")

    # ------------------------------------------------------------------
    # Validate: same inputs, both runtimes, compare outputs. Real sentences,
    # not synthetic noise, so this exercises the tokenizer/pooling/masking
    # path the same way a live request would.
    # ------------------------------------------------------------------
    print("\nvalidating: Keras vs ONNX Runtime on real inputs...")
    import onnxruntime as ort
    from lite_tokenizer import load_tokenizer, pad_sequences
    from text_cleaning import clean_text

    tok = load_tokenizer(os.path.join(ARTIFACT_DIR, "tokenizer.json"))
    test_sentences = [
        "hi",
        "i feel bad",
        "i can't stop checking the locks, it takes me an hour to leave",
        "i feel empty and terrified everyone i love will abandon me",
        "i haven't slept in three days but i feel amazing, i started four "
        "projects this week and spent money i don't really have",
        "had a good week, went hiking with friends and finally caught up on sleep",
        "my dad died in march and i still can't listen to the music he liked",
        "what time does the post office close on saturdays",
        "",  # empty input -- the all-pad edge case classifier.py handles specially
    ]

    seqs = tok.texts_to_sequences([clean_text(t) for t in test_sentences])
    X = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post",
                      dtype="int32")
    oov_id = tok.word_index.get(tok.oov_token, 1)
    all_pad = X.sum(axis=1) == 0
    X[all_pad, 0] = oov_id

    keras_out = model.predict(X, verbose=0)

    sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: X})[0]

    diff = np.abs(keras_out - onnx_out)
    max_diff = float(diff.max())
    within_tol = np.allclose(keras_out, onnx_out, atol=ATOL, rtol=RTOL)

    print(f"\n  max absolute difference across {len(test_sentences)} sentences "
          f"x {keras_out.shape[1]} classes: {max_diff:.6f}")
    print(f"  tolerance: atol={ATOL}, rtol={RTOL}")
    print(f"  {'PASS' if within_tol else 'FAIL'}")

    if not within_tol:
        os.remove(tmp_path)
        print("\nNOT PROMOTING model_v5.onnx -- outputs diverged from Keras.")
        print("The ONNX file has been deleted. classifier.py will continue")
        print("using the Keras path. Do not enable USE_ONNX until this is")
        print("investigated -- a silently wrong classifier is worse than a")
        print("slow one.")
        sys.exit(1)

    # side-by-side on a few interesting rows, for a human to eyeball
    print("\n  per-sentence top prediction, Keras vs ONNX:")
    label_map = json.load(open(os.path.join(ARTIFACT_DIR, "label_map.json")))
    id2label = {int(k): v for k, v in label_map["id_to_label"].items()}
    for i, s in enumerate(test_sentences):
        k_top = id2label[int(keras_out[i].argmax())]
        o_top = id2label[int(onnx_out[i].argmax())]
        match = "OK " if k_top == o_top else "!! "
        print(f"    {match}{s[:55]:<57} keras={k_top:<14} onnx={o_top}")

    os.replace(tmp_path, ONNX_PATH)
    print(f"\npromoted -> {ONNX_PATH}")
    print(f"total time: {time.time()-t0:.1f}s")
    print(f"\nSet USE_ONNX=true (see classifier.py) to serve from this file.")


if __name__ == "__main__":
    main()
