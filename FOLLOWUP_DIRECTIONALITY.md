# Follow-up: Temporal Directionality and Layer Localization

This is a phenomenon probe motivated by the negative backward-mean result. It does not alter the real vanilla trajectory and is not a decoding method.

## Fixed hypotheses

1. Backward fusion may fail because it moves current K/V toward older states. If the trajectory carries temporal direction, forward extrapolation may better predict the future vanilla target.
2. Any forward benefit must exceed matched random and cross-position velocity controls.
3. A logit-space forward effect must be separated from generic current-logit sharpening.
4. Temporal direction may be localized by layer even when all-layer fusion is harmful.
5. If `current - history` is temporal momentum, it should align with `reveal_state - current` at the same position more than shuffled velocity does.

## Fixed conditions

All use `H=4`, `alpha=0.25`, the original positions/schedule, and no tuning.

- Vanilla
- Backward hidden mean: `current + alpha * (history_mean - current)`
- Forward hidden mean: `current + alpha * (current - history_mean)`
- Forward hidden last-step velocity: `current + alpha * (current - previous)`
- Shuffled velocity: another eligible position's velocity, rescaled to the target velocity norm
- Random velocity: random direction matched to the target velocity norm
- Forward velocity in layers 0–7, 8–15, 16–23, or 24–31 only
- Backward logit mean
- Forward logit mean
- Forward logit last-step velocity
- Current-logit sharpening with displacement norm matched to logit velocity

The existing 200 formal vanilla outputs are immutable references. Every follow-up traced output must equal its corresponding stored output. Counterfactual conditions are evaluated sequentially from the same current state.

Norm matching records both absolute error and relative error; the numerical gate is relative error at most `2e-6`, avoiding scale-dependent rejection of large layer vectors.

## Geometry

For every selected position and layer, record only derived scalars—not K/V tensors:

- cosine between `current - history_mean` and `reveal - current`
- cosine between `current - previous` and `reveal - current`
- cosine between same-position velocity and shuffled-position velocity
- cosine between shuffled-position velocity and the target position's reveal displacement
- past/current/future displacement norms

Primary causal comparisons are Forward Mean/Velocity vs Vanilla, Random, Shuffled, Backward, and forward-logit controls. Layer-band results are localization diagnostics. All confidence intervals remain sample-clustered.
