"""Gemini API token pricing and usage extraction utilities.

Maps each Gemini model ID to per-token costs (USD) and provides helpers
to extract token usage from API responses and calculate costs.

Pricing is based on published Google AI Studio rates. Update the
GEMINI_PRICING dict when rates change or new models are added.
"""

from __future__ import annotations

# USD per token, by model ID.
# Sources: https://ai.google.dev/gemini-api/docs/pricing (March 2026)
GEMINI_PRICING: dict[str, dict[str, float]] = {
    "gemini-3-flash-preview": {
        "input":    0.50 / 1_000_000,   # $0.50 per 1M input tokens (text/image/video)
        "output":   3.00 / 1_000_000,   # $3.00 per 1M output tokens
        "thinking": 3.00 / 1_000_000,   # same as output
    },
    "gemini-3.1-flash-lite-preview": {
        "input":    0.25 / 1_000_000,   # $0.25 per 1M input tokens (text/image/video)
        "output":   1.50 / 1_000_000,   # $1.50 per 1M output tokens
        "thinking": 0.0,                # no thinking for lite
    },
    "gemini-3.1-flash-image-preview": {
        "input":    0.50 / 1_000_000,   # $0.50 per 1M input tokens (text/image)
        "output":   3.00 / 1_000_000,   # $3.00 per 1M output text/thinking tokens
        "thinking": 3.00 / 1_000_000,   # same as output
        "output_image": 60.00 / 1_000_000,  # $60.00 per 1M image output tokens
    },
}


def extract_token_usage(usage_metadata) -> dict[str, int]:
    """Extract a normalised token-usage dict from a Gemini response's usage_metadata."""
    if usage_metadata is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
    return {
        "input_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(usage_metadata, "candidates_token_count", 0) or 0,
        "thinking_tokens": getattr(usage_metadata, "thoughts_token_count", 0) or 0,
        "cached_tokens": getattr(usage_metadata, "cached_content_token_count", 0) or 0,
        "total_tokens": getattr(usage_metadata, "total_token_count", 0) or 0,
    }


def calculate_cost(model_id: str, usage: dict[str, int]) -> float:
    """Return the estimated USD cost for a single API call's token usage.

    For image-generating models (those with an 'output_image' rate), output
    tokens are treated as image output tokens since the API reports them
    together under candidates_token_count.
    """
    pricing = GEMINI_PRICING.get(model_id)
    if not pricing:
        return 0.0
    output_rate = pricing.get("output_image", pricing.get("output", 0))
    return (
        usage.get("input_tokens", 0) * pricing.get("input", 0)
        + usage.get("output_tokens", 0) * output_rate
        + usage.get("thinking_tokens", 0) * pricing.get("thinking", 0)
    )


def calculate_cost_for_log(model_id: str, records: list[dict]) -> float:
    """Sum cost across a list of token-usage records for a given model."""
    return sum(calculate_cost(model_id, r) for r in records)


def format_usage_summary(records: list[dict]) -> dict:
    """Aggregate a list of token-usage records into a summary with totals and cost."""
    by_model: dict[str, dict[str, int]] = {}
    for r in records:
        mid = r.get("model", "unknown")
        if mid not in by_model:
            by_model[mid] = {
                "input_tokens": 0, "output_tokens": 0,
                "thinking_tokens": 0, "cached_tokens": 0,
                "total_tokens": 0, "calls": 0,
            }
        for k in ("input_tokens", "output_tokens", "thinking_tokens", "cached_tokens", "total_tokens"):
            by_model[mid][k] += r.get(k, 0)
        by_model[mid]["calls"] += 1

    total_cost = 0.0
    for mid, totals in by_model.items():
        cost = calculate_cost(mid, totals)
        totals["cost_usd"] = round(cost, 6)
        total_cost += cost

    grand = {
        "input_tokens": sum(v["input_tokens"] for v in by_model.values()),
        "output_tokens": sum(v["output_tokens"] for v in by_model.values()),
        "thinking_tokens": sum(v["thinking_tokens"] for v in by_model.values()),
        "cached_tokens": sum(v["cached_tokens"] for v in by_model.values()),
        "total_tokens": sum(v["total_tokens"] for v in by_model.values()),
        "calls": sum(v["calls"] for v in by_model.values()),
        "cost_usd": round(total_cost, 6),
    }

    return {"by_model": by_model, "grand_total": grand}
