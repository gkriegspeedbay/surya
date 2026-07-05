"""Small cross-version (transformers 4.x / 5.x) compat helpers.

Kept in one place because the same needs (recompute a non-persistent buffer
during _init_weights, detect the major version, polyfill a removed pytorch_utils
helper) recur across several model files. See huggingface/transformers#46620
and https://github.com/datalab-to/surya/issues/492.
"""
import torch

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


def transformers_major_version() -> int:
    """First dot-separated segment of transformers.__version__, as an int.
    Assumes a plain "X.Y.Z[...]" version string (true for every transformers
    release to date); a non-numeric leading segment would raise ValueError,
    same as the ad-hoc int(...) calls this replaces used to."""
    import transformers

    return int(transformers.__version__.split(".")[0])


def is_transformers_5_plus() -> bool:
    return transformers_major_version() >= 5


try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices
except ImportError:
    # transformers >= 5.0 removed this helper from pytorch_utils
    # (https://github.com/datalab-to/surya/issues/492). Vendor the
    # historical implementation so head pruning keeps working under 5.x.
    # Only invoked if a *.prune_heads() method is called -- which surya
    # inference never does -- but the import must resolve for callers to
    # load at all.
    from typing import List as _List, Set as _Set

    def find_pruneable_heads_and_indices(
        heads: _List[int], n_heads: int, head_size: int, already_pruned_heads: _Set[int]
    ):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index
