from collections import Counter


def count_trainable_params(model):
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / max(total, 1),
    }


def summarize_ranks(rank_pattern):
    ranks = [int(rank) for rank in rank_pattern.values()]
    if not ranks:
        return {}
    histogram = Counter(ranks)
    return {
        "num_modules": len(ranks),
        "min_rank": min(ranks),
        "max_rank": max(ranks),
        "avg_rank": sum(ranks) / len(ranks),
        "rank_histogram": {str(rank): histogram[rank] for rank in sorted(histogram)},
    }
