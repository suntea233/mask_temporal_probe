from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class ProjectedState:
    k: dict[int, torch.Tensor]
    v: dict[int, torch.Tensor]
    h: dict[int, torch.Tensor]


class ProjectionController:
    """Observe or replace LLaDA K/V projection outputs before RoPE."""

    def __init__(self, model):
        modules = dict(model.named_modules())
        k_modules = {name[:-7]: mod for name, mod in modules.items() if name.endswith(".k_proj")}
        v_modules = {name[:-7]: mod for name, mod in modules.items() if name.endswith(".v_proj")}
        # Preserve module traversal order (physical Transformer order); lexical
        # sorting would incorrectly place layer 10 before layer 2.
        prefixes = [name[:-7] for name in modules if name.endswith(".k_proj") and name[:-7] in v_modules]
        if not prefixes:
            raise RuntimeError("No paired k_proj/v_proj modules found")
        self.layer_names = prefixes
        self.block_modules = [modules[prefix] for prefix in prefixes]
        self._handles = []
        self._positions: list[int] = []
        self._capture_kv = False
        self._capture_h = False
        self._batch_indices: list[int] = []
        self._replacements: dict[str, dict[int, torch.Tensor]] = {"k": {}, "v": {}, "h": {}}
        self.captured: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
        for layer, prefix in enumerate(prefixes):
            self._handles.append(k_modules[prefix].register_forward_hook(self._hook("k", layer)))
            self._handles.append(v_modules[prefix].register_forward_hook(self._hook("v", layer)))
            self._handles.append(self.block_modules[layer].register_forward_hook(self._hidden_hook(layer)))

    @property
    def n_layers(self) -> int:
        return len(self.layer_names)

    def _hook(self, kind: str, layer: int):
        def hook(_module, _inputs, output):
            replacement = self._replacements[kind].get(layer)
            if replacement is not None:
                output = output.clone()
                output[self._batch_indices, self._positions, :] = replacement.to(output.device, output.dtype)
            if self._capture_kv and self._positions:
                self.captured[kind][layer] = output[0, self._positions, :].detach().cpu().clone()
            return output
        return hook

    def _hidden_hook(self, layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            replacement = self._replacements["h"].get(layer)
            if replacement is not None:
                hidden = hidden.clone()
                hidden[self._batch_indices, self._positions, :] = replacement.to(hidden.device, hidden.dtype)
                output = (hidden, *output[1:]) if isinstance(output, tuple) else hidden
            if self._capture_h and self._positions:
                self.captured["h"][layer] = hidden[0, self._positions, :].detach().cpu().clone()
            return output
        return hook

    @contextmanager
    def mode(
        self,
        positions: list[int],
        *,
        capture: bool = False,
        capture_kv: bool | None = None,
        capture_h: bool | None = None,
        batch_indices: list[int] | None = None,
        k: dict[int, torch.Tensor] | None = None,
        v: dict[int, torch.Tensor] | None = None,
        h: dict[int, torch.Tensor] | None = None,
    ) -> Iterator[None]:
        if self._positions or self._capture_kv or self._capture_h or any(self._replacements[kind] for kind in ("k", "v", "h")):
            raise RuntimeError("ProjectionController modes cannot be nested")
        self._positions = list(positions)
        self._batch_indices = list(batch_indices) if batch_indices is not None else [0] * len(self._positions)
        if len(self._batch_indices) != len(self._positions):
            raise ValueError("batch_indices and positions must have equal length")
        self._capture_kv = capture if capture_kv is None else capture_kv
        self._capture_h = capture if capture_h is None else capture_h
        self._replacements = {"k": k or {}, "v": v or {}, "h": h or {}}
        self.captured = defaultdict(dict)
        try:
            yield
        finally:
            self._positions = []
            self._batch_indices = []
            self._capture_kv = False
            self._capture_h = False
            self._replacements = {"k": {}, "v": {}, "h": {}}

    def state(self) -> ProjectedState:
        if any(len(self.captured[kind]) != self.n_layers for kind in ("k", "v", "h")):
            raise RuntimeError("Incomplete H/K/V capture")
        return ProjectedState(dict(self.captured["k"]), dict(self.captured["v"]), dict(self.captured["h"]))

    def hidden_state(self) -> dict[int, torch.Tensor]:
        if len(self.captured["h"]) != self.n_layers:
            raise RuntimeError("Incomplete hidden-state capture")
        return dict(self.captured["h"])

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def matched_random(displacement: torch.Tensor, seed: int) -> torch.Tensor:
    """Independent Gaussian direction with exactly matched per-row L2 norm."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    random = torch.randn(displacement.shape, generator=generator, dtype=torch.float32)
    target = displacement.float().norm(dim=-1, keepdim=True)
    norm = random.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return (random * target / norm).to(displacement.dtype)
