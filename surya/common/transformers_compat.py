"""Small cross-version (transformers 4.x / 5.x) compat helpers.

Kept in one place because the same need (recompute a non-persistent buffer
during _init_weights, on both transformers versions) recurs across several
model files. See huggingface/transformers#46620.
"""
try:
    from transformers import initialization as _hf_init

    def copy_(dst, value):
        # transformers >= 5.0: dst may be an uninitialized meta tensor (no
        # backing storage) after a meta-device from_pretrained -- plain
        # tensor.copy_() can't target that, so use the framework's own
        # helper, which knows how to materialize it first.
        _hf_init.copy_(dst, value)

except ImportError:

    def copy_(dst, value):
        # transformers < 5.0 never meta-device-initializes these buffers in
        # the first place -- dst already has real backing storage with its
        # __init__-computed value, so a plain in-place copy is equivalent
        # (and this whole call is a harmless no-op re-assignment there).
        dst.copy_(value)
