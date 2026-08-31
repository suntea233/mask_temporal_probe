from types import SimpleNamespace

import torch
from torch import nn

from src.config import ProbeConfig
from src.temporal_sampler import traced_generate


class ToyBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return x + torch.tanh(self.k_proj(x) + self.v_proj(x))


class ToyDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.embedding = nn.Embedding(11, 6)
        self.blocks = nn.ModuleList([ToyBlock(6), ToyBlock(6)])
        self.head = nn.Linear(6, 11, bias=False)
        self.config = SimpleNamespace(eos_token_id=9)

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        # Let revealed context alter every position, as non-causal attention would.
        x = x + x.mean(dim=1, keepdim=True)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        logits[..., 9:] = -100
        return SimpleNamespace(logits=logits)


def test_tiny_end_to_end_probe_is_deterministic_and_keeps_kv_off_records():
    model = ToyDiffusion().eval()
    config = ProbeConfig(
        steps=4, gen_length=4, block_length=4, history=1, n_mask=2,
        progress_fractions=(0.5,), mask_id=10,
    )
    prompt = torch.tensor([[1]])
    first, records, sanity = traced_generate(
        model, prompt, torch.ones_like(prompt), config,
        sample_id=0, special_token_ids={9, 10},
    )
    second, records_2, sanity_2 = traced_generate(
        model, prompt, torch.ones_like(prompt), config,
        sample_id=0, special_token_ids={9, 10},
    )
    assert torch.equal(first, second)
    assert records == records_2
    assert sanity == sanity_2
    assert sanity["projection_layers"] == 2
    assert sanity["vanilla_probe_max_abs_logit_error"] == 0
    assert records
    assert all(record["absolute_position"] != record["shuffle_source_position"] for record in records)
    assert "condition_logits" not in records[0]
    assert "trajectory_logits" not in records[0]
