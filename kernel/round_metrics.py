"""Round-level metrics computation for the WorldKernel."""

from typing import Dict, List


def compute_gini(values: List[float]) -> float:
    """Compute the Gini coefficient of a list of values."""
    n = len(values)
    if n <= 1:
        return 0.0
    total = sum(values)
    if total == 0.0:
        return 0.0
    sorted_vals = sorted(values)
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        weighted_sum += (2 * (i + 1) - n - 1) * v
    return weighted_sum / (n * total)


def compute_round_metrics(
    members: List[Dict],
    trade_volume: int = 0,
    conflict_count: int = 0,
    mechanism_proposals: int = 0,
    mechanism_approvals: int = 0,
) -> Dict[str, float]:
    """Compute aggregate metrics for a round."""
    vitalities = [m.get("vitality", 0.0) for m in members]
    return {
        "total_vitality": sum(vitalities),
        "gini_coefficient": compute_gini(vitalities),
        "trade_volume": trade_volume,
        "conflict_count": conflict_count,
        "mechanism_proposals": mechanism_proposals,
        "mechanism_approvals": mechanism_approvals,
        "population": len(members),
    }
