"""
model_layers.py
===============
Custom Keras layers the saved model depends on.

WHY THIS FILE EXISTS
--------------------
`@register_keras_serializable` only takes effect when the decorated class is
IMPORTED -- the decorator has to actually run before `load_model` looks up the
name. The class lives in the training notebook, so at serving time Keras finds
"therapygpt>MaskedGlobalPooling1D" in the saved config and has nothing to build
from:

    Could not locate class 'MaskedGlobalPooling1D'

So the layer is redefined here, byte-identical in behaviour to the training
version, and imported by classifier.py before the model is loaded.

WHAT THE LAYER DOES, AND WHY IT IS NOT STOCK
--------------------------------------------
It takes [features, tokens] and returns [masked max-over-time, masked
mean-over-time].

Stock GlobalMaxPooling1D does not reliably honour a propagated mask, and Keras
RNNs default to zero_output_for_mask=False -- so padded timesteps hold a REPEAT
of the last valid output rather than zeros. Pooling over them is silently
wrong: no error, slightly worse model. This computes the mask from the token
ids directly instead of trusting propagation.

If you change anything here, the model's behaviour changes. Don't.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

# Keras 3 moved the decorator; Keras 2 still has it under utils.
try:
    _register = keras.saving.register_keras_serializable
except AttributeError:  # pragma: no cover
    _register = keras.utils.register_keras_serializable


@_register(package="therapygpt")
class MaskedGlobalPooling1D(keras.layers.Layer):
    """Concatenate [masked max-over-time, masked mean-over-time] -> (B, 2F)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        feats, tokens = inputs
        m = tf.cast(tf.not_equal(tokens, 0), feats.dtype)[..., tf.newaxis]

        # -1e4 rather than -1e9: under mixed_float16 anything past ~65504
        # overflows to -inf and poisons the max. LSTM outputs are tanh-bounded
        # to [-1, 1], so -1e4 is already an unreachable floor.
        neg = tf.cast(-1e4, feats.dtype)
        mx = tf.reduce_max(tf.where(m > 0, feats, neg), axis=1)

        mean = tf.reduce_sum(feats * m, 1) / tf.maximum(
            tf.reduce_sum(m, 1), tf.cast(1e-9, feats.dtype))

        # A row with NO real tokens leaves mx at the -1e4 sentinel, which the
        # next Dense sums over 256 features -> overflow -> NaN. Zero those rows
        # so the sentinel can never leave this layer.
        has_any = tf.reduce_max(m, axis=1)
        return tf.concat([mx * has_any, mean * has_any], -1)

    def compute_mask(self, inputs, mask=None):
        return None          # the time dimension is gone

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1] * 2)


CUSTOM_OBJECTS = {
    "MaskedGlobalPooling1D": MaskedGlobalPooling1D,
    "therapygpt>MaskedGlobalPooling1D": MaskedGlobalPooling1D,
}
