from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...]) -> list[torch.Tensor]:
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    scores = []
    for k in topk:
        k = min(k, logits.shape[1])
        correct_k = correct[:k].reshape(-1).float().sum()
        scores.append(correct_k * (100.0 / target.numel()))
    return scores

