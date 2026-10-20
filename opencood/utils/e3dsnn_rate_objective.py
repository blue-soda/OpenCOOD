"""Differentiable uncompressed sparse payload proxy, not measured NPZ bytes."""
import torch


def sparse_payload_proxy(counts, source_rows):
    """80 address/mask bits per active row plus two bits per nonzero count.

    Forward is the integer sparse payload size, normalized by the maximum
    144 bits per selected 32-channel source row. Backward uses bounded smooth
    occupancy. Excludes headers, zlib, and choice of dense versus sparse codec.
    """
    if counts.ndim != 5 or counts.shape[:2] != (1, 32):
        raise ValueError('Expected one 32-channel Count3 volume')
    if (not torch.isfinite(counts).all() or
            torch.any((counts < 0) | (counts > 3) | (counts != counts.floor()))):
        raise ValueError('Expected exact Count3 values')
    if type(source_rows) is not int or source_rows < 0:
        raise ValueError('Expected nonnegative selected source row count')
    active_rows = (counts.detach() != 0).any(1).sum()
    if active_rows > source_rows:
        raise ValueError('Content support exceeds selected source support')
    soft = counts / (counts + 1.)
    occupancy = (counts.detach() > 0).to(counts.dtype) + (soft - soft.detach())
    row_count = counts.sum(1)
    row_soft = row_count / (row_count + 1.)
    row_occupancy = (row_count.detach() > 0).to(counts.dtype) + (row_soft-row_soft.detach())
    return (80*row_occupancy.sum() + 2*occupancy.sum()) / (144*max(source_rows, 1))
