import torch
from torch import nn

from src.projection import ProjectionController, matched_random


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.k_proj(x) + self.v_proj(x)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_capture_and_only_selected_replacement():
    torch.manual_seed(2)
    model, x = Tiny(), torch.randn(1, 3, 4)
    controller = ProjectionController(model)
    original = model(x)
    with controller.mode([1], capture=True):
        assert torch.equal(model(x), original)
    state = controller.state()
    replacement = {layer: torch.zeros_like(value) for layer, value in state.k.items()}
    with controller.mode([1], k=replacement):
        changed = model(x)
    assert torch.equal(changed[:, 0], original[:, 0])
    assert torch.equal(changed[:, 2], original[:, 2])
    assert not torch.equal(changed[:, 1], original[:, 1])


def test_random_norm_is_matched_per_row_and_reproducible():
    displacement = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    first = matched_random(displacement, 17)
    second = matched_random(displacement, 17)
    assert torch.equal(first, second)
    assert torch.allclose(first.norm(dim=-1), displacement.norm(dim=-1), atol=1e-6)


def test_batched_hidden_replacement_isolated_by_batch_and_position():
    torch.manual_seed(3)
    model = Tiny()
    x = torch.randn(2, 3, 4)
    controller = ProjectionController(model)
    original = model(x)
    replacements = torch.zeros(2, 4)
    with controller.mode([0, 2], batch_indices=[0, 1], h={0: replacements}):
        changed = model(x)
    assert not torch.equal(changed[0, 0], original[0, 0])
    assert torch.equal(changed[0, 1:], original[0, 1:])
    assert torch.equal(changed[1, :2], original[1, :2])
    assert not torch.equal(changed[1, 2], original[1, 2])
    controller.close()
