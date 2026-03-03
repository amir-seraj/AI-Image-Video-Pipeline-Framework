"""Debug VLM judge prompts interactively.

Loads a VLM, sends it a reference shoe image + generated candidate image
along with a prompt, and streams the response. Edit PROMPT below and
re-run to iterate on prompt wording.

Usage:
    python scripts/debug_vlm_judge.py --shoe tests/Image/shoes001.jpeg --candidate path/to/generated.png
    python scripts/debug_vlm_judge.py --shoe tests/Image/shoes001.jpeg --candidate path/to/generated.png --vlm 8b-thinking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Default features (override with --features) ─────────────────────────
DEFAULT_FEATURES = [
    "shoe type and silhouette",
    "color and pattern",
    "heel type and height",
    "material and texture",
    "toe shape",
    "closure and straps",
    "sole style",
    "both feet match",
]

# ── Edit this prompt and re-run ──────────────────────────────────────────
PROMPT = """\
You are a strict quality inspector comparing shoes across two images.

The first image is the REFERENCE SHOE — a product photo of the target shoe.
The second image shows a PERSON WEARING SHOES — examine only the shoes \
on their feet.

Your job: for each listed attribute, score how closely the shoes on the \
person's feet match the same attribute in the REFERENCE SHOE.

IMPORTANT: The shoes on the person are AI-generated replacements. They \
often look correct at first glance but have subtle differences in \
proportions, trim details, strap attachment points, edge finishing, or \
color shade. Examine each attribute at the fine-detail level, not just \
the category level. "Both are red" does not earn a 5 — check the exact \
shade, glossiness, and reflections.

{iteration_context}\
Scoring scale:
  1 = completely different from the reference
  2 = vaguely similar but clearly wrong
  3 = same general type but noticeable differences in detail
  4 = close match with only minor differences visible on close inspection
  5 = indistinguishable — identical shape, proportions, color shade, and \
every visible detail

A score of 5 means you cannot find ANY difference for that attribute. \
If you can spot even one small difference, the score must be 4 or lower.

Attributes to score: {features}

Check BOTH feet individually. For "both feet match": score 1 if the two \
feet show different shoes (one matches the reference, the other does not); \
score 5 only if both feet wear identical shoes that match the REFERENCE SHOE.

First, describe what you observe in the reference shoe and on the \
person's feet for each attribute. Then provide your scores.

Reply in this exact format:
OBSERVATIONS: <For each attribute, state what you see in the reference \
vs. what you see on the person's feet. Note any differences no matter \
how small.>
SCORES: {score_format}
REPAIR: <For each attribute scored below 5, describe the specific \
mismatch between what you see on the person's feet and the reference.>

Example of a correctly filled response (scores are illustrative):
SCORES: {example_format}
REPAIR: The heel appears lower than the reference ...
"""
# ─────────────────────────────────────────────────────────────────────────

VLM_MODELS = {
    "8b": "qwen3_vl_8b",
    "8b-thinking": "qwen3_vl_8b_thinking",
    "30b": "qwen3_vl_30b",
}


def main():
    parser = argparse.ArgumentParser(description="Debug VLM judge prompt")
    parser.add_argument("--shoe", required=True, help="Path to reference shoe image")
    parser.add_argument("--candidate", required=True, help="Path to generated/candidate image")
    parser.add_argument(
        "--vlm", choices=list(VLM_MODELS), default="8b",
        help="VLM variant (default: 8b)",
    )
    parser.add_argument(
        "--features", nargs="+", default=None,
        help="Override feature list (default: use DEFAULT_FEATURES)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=10240,
        help="Max new tokens (default: 10240)",
    )
    args = parser.parse_args()

    import torch
    from PIL import Image as PILImage
    from casadei.media import ImageMedia, TextMedia, MediaBundle
    from casadei.models.registry import default_registry

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    # Resolve features and format the prompt
    features = args.features or DEFAULT_FEATURES
    features_str = ", ".join(features)
    score_format = ", ".join(f"{f}=[1-5]" for f in features)
    example_format = ", ".join(f"{f}=3" for f in features)

    prompt = PROMPT.format(
        features=features_str,
        score_format=score_format,
        example_format=example_format,
        iteration_context="This is comparison 1 of 5.\n",
    )

    # Load images
    shoe_img = PILImage.open(args.shoe).convert("RGB")
    candidate_img = PILImage.open(args.candidate).convert("RGB")
    print(f"Shoe:      {args.shoe}  ({shoe_img.size[0]}x{shoe_img.size[1]})")
    print(f"Candidate: {args.candidate}  ({candidate_img.size[0]}x{candidate_img.size[1]})")
    print(f"VLM:       {VLM_MODELS[args.vlm]}")
    print(f"Features:  {features}")
    print(f"Prompt:\n{prompt}")
    print("=" * 60)

    # Load model
    model_cls = default_registry.get(VLM_MODELS[args.vlm])
    model = model_cls()
    print("Loading model...")
    model.load_model()

    # Build bundle — images go in insertion order: shoe first, candidate second
    bundle = MediaBundle(items={
        "reference": ImageMedia(image=shoe_img),
        "candidate": ImageMedia(image=candidate_img),
        "prompt": TextMedia(text=prompt),
    })

    # Override max_new_tokens for this run
    model.DEFAULT_PARAMS = {**model.DEFAULT_PARAMS, "max_new_tokens": args.max_tokens}

    # Stream response
    print("\n--- VLM Response ---")
    chunks: list[str] = []
    for chunk in model.run_streaming(bundle):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        chunks.append(chunk)
    print("\n--- End ---")

    # Cleanup
    model.unload_model()
    torch.cuda.empty_cache()

    full_response = "".join(chunks).strip()
    print(f"\nTokens (approx): {len(full_response.split())}")


if __name__ == "__main__":
    main()
