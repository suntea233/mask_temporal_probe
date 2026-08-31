from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProbeConfig:
    model_id: str = "GSAI-ML/LLaDA-8B-Instruct"
    model_revision: str = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
    dataset_id: str = "gsm8k"
    dataset_revision: str = "740312add88f781978c0658806c59bc2815b9866"
    dataset_config: str = "main"
    split: str = "test"
    samples: int = 200
    steps: int = 128
    gen_length: int = 256
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = 126336
    history: int = 4
    alpha: float = 0.25
    n_mask: int = 4
    progress_fractions: tuple[float, ...] = (0.25, 0.50, 0.75)
    seed: int = 20260827

    def as_dict(self) -> dict:
        return asdict(self)

