"""Physical layer shrinking after structured pruning.

After torch.nn.utils.prune.ln_structured() + prune.remove(), pruned output
channels remain as zero rows in the weight matrix.  The layer is still the
original (dense) shape — no speedup.  This module physically removes those
dead rows and their corresponding input columns in the next layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def shrink_linear_pair(
    parent: nn.Linear, child: nn.Linear
) -> tuple[nn.Linear, nn.Linear]:
    """Physically remove pruned output channels from parent and matching input
    columns from child.

    Pre-condition: parent.out_features == child.in_features.
    A row in parent.weight is considered pruned if its L1 norm is zero.

    Returns the original pair unchanged if nothing was pruned.
    """
    if parent.out_features != child.in_features:
        raise ValueError(
            f"Dimension mismatch: parent.out={parent.out_features}, "
            f"child.in={child.in_features}"
        )

    active = parent.weight.data.abs().sum(dim=1) > 0  # (out_parent,)
    n_active = int(active.sum().item())

    if n_active == parent.out_features:
        return parent, child

    dev = parent.weight.device
    dtype = parent.weight.dtype

    new_parent = nn.Linear(
        parent.in_features, n_active,
        bias=parent.bias is not None, device=dev, dtype=dtype,
    )
    new_parent.weight.data = parent.weight.data[active]
    if parent.bias is not None:
        new_parent.bias.data = parent.bias.data[active]

    new_child = nn.Linear(
        n_active, child.out_features,
        bias=child.bias is not None, device=dev, dtype=dtype,
    )
    new_child.weight.data = child.weight.data[:, active]
    if child.bias is not None:
        new_child.bias.data = child.bias.data.clone()

    return new_parent, new_child


def _set_module(root: nn.Module, dotted_name: str, new_mod: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_mod)


def shrink_sequential_linears(model: nn.Module) -> int:
    """Physically shrink pruned Linear pairs that are directly sequential.

    Only processes pairs that sit inside the same nn.Sequential container —
    this guarantees they are truly consecutive in the computation graph.
    Pairs scattered across different sub-modules are skipped; they may share
    a dimension by coincidence but are not directly connected.

    Covers: MLP classifier heads, DLinear projection stack, feed-forward blocks.
    Skips: GRU recurrent weights, graph attention, SSM state projections, any
           Linear pair that is not inside an nn.Sequential.

    Returns the number of pairs shrunk.
    """
    shrunk = 0

    for seq_name, seq_mod in model.named_modules():
        if not isinstance(seq_mod, nn.Sequential):
            continue

        # Collect (index_in_sequential, Linear) pairs in order
        linears = [
            (idx, child)
            for idx, child in enumerate(seq_mod)
            if isinstance(child, nn.Linear)
        ]

        processed_as_child: set[int] = set()
        for k in range(len(linears) - 1):
            idx1, m1 = linears[k]
            idx2, m2 = linears[k + 1]

            if idx1 in processed_as_child:
                continue
            if m1.out_features != m2.in_features:
                continue

            zero_rows = int((m1.weight.data.abs().sum(dim=1) == 0).sum())
            if zero_rows == 0:
                continue

            new_m1, new_m2 = shrink_linear_pair(m1, m2)
            seq_mod[idx1] = new_m1
            seq_mod[idx2] = new_m2
            processed_as_child.add(idx2)
            shrunk += 1

    return shrunk
