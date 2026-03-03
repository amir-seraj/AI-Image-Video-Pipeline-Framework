"""Debug VLM judge with multiple specialized agents.

Splits the monolithic judge prompt into 4 domain-specific agents that each
focus on a narrow set of attributes. This forces deeper per-domain analysis
instead of lazy category-level matching ("both are red platform sandals = 5").

The model loads once; each agent runs sequentially on the single GPU.

Usage:
    python scripts/debug_vlm_multi_judge.py \
        --shoe tests/Image/shoes001.jpeg \
        --candidate path/to/generated.png

    python scripts/debug_vlm_multi_judge.py \
        --shoe tests/Image/shoes001.jpeg \
        --candidate path/to/generated.png \
        --vlm 8b-thinking
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Specialized Agent Definitions
# ---------------------------------------------------------------------------

_AGENT_PROMPT_TEMPLATE = """\
You are an expert {domain} inspector comparing shoes across two images.

The first image is the REFERENCE SHOE — a product photo of the target shoe.
The second image shows a PERSON WEARING SHOES — examine only the shoes \
on their feet.

{domain_instructions}

IMPORTANT: The shoes on the person are AI-generated replacements. They \
often look correct at first glance but have subtle differences. Examine \
each attribute at the fine-detail level, not just the category level.

Scoring scale:
  1 = completely different from the reference
  2 = vaguely similar but clearly wrong
  3 = same general type but noticeable differences in detail
  4 = close match with only minor differences visible on close inspection
  5 = indistinguishable — identical in every visible detail

A score of 5 means you cannot find ANY difference for that attribute. \
If you can spot even one small difference, the score must be 4 or lower.

Attributes to score: {features}

First, describe what you observe in the reference shoe and on the \
person's feet for each attribute. Then provide your scores.

Reply in this exact format:
OBSERVATIONS: <For each attribute, state what you see in the reference \
vs. what you see on the person's feet. Note any differences no matter \
how small.>
SCORES: {score_format}
REPAIR: <For each attribute scored below 5, describe the specific \
mismatch between what you see on the person's feet and the reference. \
If all scores are 5, write "No issues.">

Example of a correctly filled response (scores are illustrative):
SCORES: {example_format}
REPAIR: The shade is slightly warmer than the reference ...
"""


@dataclass
class AgentSpec:
    name: str
    domain: str
    domain_instructions: str
    features: list[str]


AGENTS: list[AgentSpec] = [
    AgentSpec(
        name="Color & Material",
        domain="color and material",
        domain_instructions=(
            "Your expertise is COLOR, PATTERN, MATERIAL, and SURFACE FINISH.\n\n"
            "Focus on:\n"
            "- Exact color shade (not just 'red' — is it cherry red, wine red, coral?)\n"
            "- Color distribution and gradients across the shoe\n"
            "- Pattern accuracy (print scale, orientation, placement)\n"
            "- Material type (leather, suede, patent, fabric, synthetic)\n"
            "- Surface texture (grain, weave, embossing depth)\n"
            "- Glossiness and reflections (matte vs satin vs high-gloss)\n"
            "- Transparency or translucency if applicable\n\n"
            "Compare each of these at the pixel level. 'Both are red' is NOT "
            "sufficient — check whether the exact shade, saturation, and "
            "surface sheen match."
        ),
        features=[
            "color shade accuracy",
            "pattern fidelity",
            "material type",
            "texture and grain",
            "glossiness and reflections",
        ],
    ),
    AgentSpec(
        name="Shape & Structure",
        domain="shape and structural",
        domain_instructions=(
            "Your expertise is SILHOUETTE, PROPORTIONS, and STRUCTURAL GEOMETRY.\n\n"
            "Focus on:\n"
            "- Overall silhouette and profile shape\n"
            "- Heel type (stiletto, block, wedge, kitten, flat)\n"
            "- Heel height relative to the rest of the shoe\n"
            "- Heel angle and taper\n"
            "- Toe box shape (pointed, round, square, almond, open)\n"
            "- Toe box proportions (width, length, depth)\n"
            "- Sole thickness and platform height\n"
            "- Arch curve and instep line\n\n"
            "Compare proportions and angles precisely. A heel that is "
            "slightly too short or a toe box that is slightly too wide "
            "should score below 5."
        ),
        features=[
            "silhouette and profile",
            "heel type and height",
            "heel angle and shape",
            "toe shape and proportions",
            "sole and platform",
        ],
    ),
    AgentSpec(
        name="Details & Hardware",
        domain="detail and hardware",
        domain_instructions=(
            "Your expertise is SMALL DETAILS, HARDWARE, and DECORATIVE ELEMENTS.\n\n"
            "Focus on:\n"
            "- Closure type (zip, buckle, lace, slip-on, velcro)\n"
            "- Strap count, width, placement, and routing\n"
            "- Buckle/clasp shape, size, color, and placement\n"
            "- Metal hardware finish (gold, silver, gunmetal, brushed, polished)\n"
            "- Decorative trim (chains, studs, crystals, bows, logos)\n"
            "- Edge finishing (piping, stitching, raw-cut, rolled)\n"
            "- Seam lines and construction details\n"
            "- Any brand markings or logos visible\n\n"
            "These small elements are where AI-generated shoes most often "
            "differ from the reference. Examine attachment points, hardware "
            "count, and decorative placement carefully."
        ),
        features=[
            "closure and straps",
            "buckles and clasps",
            "hardware finish and color",
            "decorative trim",
            "edge finishing and seams",
        ],
    ),
    AgentSpec(
        name="Consistency",
        domain="consistency and realism",
        domain_instructions=(
            "Your expertise is LEFT-RIGHT CONSISTENCY and PLACEMENT REALISM.\n\n"
            "Focus on:\n"
            "- Do BOTH feet show the same shoe? Compare left foot vs right foot.\n"
            "- Are both shoes the same color, shape, and style?\n"
            "- Do the shoes sit naturally on the person's feet?\n"
            "- Is the scale/size proportionate to the person's body?\n"
            "- Do the shoes make proper contact with the ground/surface?\n"
            "- Are there any impossible geometry artifacts (floating, clipping)?\n"
            "- Do shadows and lighting on the shoes match the scene?\n\n"
            "AI-generated replacements often get one foot right and the other "
            "wrong, or produce shoes that float or clip through the ground."
        ),
        features=[
            "both feet match",
            "natural foot placement",
            "scale and proportion to body",
            "ground contact realism",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Score Parsing (reused from judge.py)
# ---------------------------------------------------------------------------

def _parse_scores(text: str) -> dict[str, float]:
    """Parse 'SCORES: attr1=4, attr2=2, ...' from VLM response."""
    scores_match = re.search(r"SCORES:\s*(.+)", text, re.IGNORECASE)
    if not scores_match:
        raise ValueError("No 'SCORES:' line found")
    scores_text = scores_match.group(1).strip()
    if re.search(r"=\[1-5\]", scores_text) or re.search(r"=N\b", scores_text):
        raise ValueError(
            "Scores contain unfilled placeholders ([1-5] or N). "
            "Replace each placeholder with an actual integer 1-5."
        )
    scores = {}
    for pair in re.split(r",\s*", scores_text):
        pair = pair.strip()
        if not pair:
            continue
        m = re.match(r"(.+?)\s*=\s*([0-9]+(?:\.[0-9]+)?)", pair)
        if m:
            scores[m.group(1).strip()] = float(m.group(2))
    if not scores:
        raise ValueError(f"Could not parse any scores from: {scores_text}")
    return scores


def _parse_repair(text: str) -> str:
    """Extract REPAIR text from VLM response."""
    repair_match = re.search(r"REPAIR:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if repair_match:
        return repair_match.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Agent Result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    agent_name: str
    scores: dict[str, float]
    repair: str
    raw_response: str
    parse_error: str | None = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VLM_MODELS = {
    "8b": "qwen3_vl_8b",
    "8b-thinking": "qwen3_vl_8b_thinking",
    "30b": "qwen3_vl_30b",
}

_MAX_RETRIES = 1  # one retry per agent on parse failure


def _run_agent(
    model,
    agent: AgentSpec,
    shoe_img,
    candidate_img,
    ImageMedia,
    TextMedia,
    MediaBundle,
) -> AgentResult:
    """Run a single specialized agent and return its result."""
    features_str = ", ".join(agent.features)
    score_format = ", ".join(f"{f}=[1-5]" for f in agent.features)
    example_format = ", ".join(f"{f}=3" for f in agent.features)

    prompt = _AGENT_PROMPT_TEMPLATE.format(
        domain=agent.domain,
        domain_instructions=agent.domain_instructions,
        features=features_str,
        score_format=score_format,
        example_format=example_format,
    )

    bundle = MediaBundle(items={
        "reference": ImageMedia(image=shoe_img),
        "candidate": ImageMedia(image=candidate_img),
        "prompt": TextMedia(text=prompt),
    })

    last_error = ""
    raw_response = ""

    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0 and last_error:
            retry_prompt = (
                f"{prompt}\n\n"
                f"Your previous response could not be parsed ({last_error}).\n"
                f"You MUST replace every [1-5] with an actual integer. "
                f"Do not output [1-5] literally.\n"
                f"Correct example:\n"
                f"SCORES: {example_format}\n"
                f"REPAIR: ..."
            )
            bundle = MediaBundle(items={
                "reference": ImageMedia(image=shoe_img),
                "candidate": ImageMedia(image=candidate_img),
                "prompt": TextMedia(text=retry_prompt),
            })

        # Stream response
        sys.stdout.write(f"\n  [{agent.name}] ")
        sys.stdout.flush()
        chunks: list[str] = []
        for chunk in model.run_streaming(bundle):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            chunks.append(chunk)
        sys.stdout.write("\n")
        sys.stdout.flush()

        raw_response = "".join(chunks).strip()

        try:
            scores = _parse_scores(raw_response)
            repair = _parse_repair(raw_response)
            return AgentResult(
                agent_name=agent.name,
                scores=scores,
                repair=repair,
                raw_response=raw_response,
            )
        except ValueError as e:
            last_error = str(e)
            print(f"  [{agent.name}] Parse error: {last_error} "
                  f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})")

    return AgentResult(
        agent_name=agent.name,
        scores={},
        repair="",
        raw_response=raw_response,
        parse_error=last_error,
    )


def _print_results(results: list[AgentResult]) -> None:
    """Print a combined results table across all agents."""
    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)

    all_scores: dict[str, float] = {}
    agent_avgs: list[tuple[str, float]] = []

    for result in results:
        if result.parse_error:
            print(f"\n  [{result.agent_name}] PARSE FAILED: {result.parse_error}")
            continue

        agent_scores = result.scores
        avg = sum(agent_scores.values()) / len(agent_scores) if agent_scores else 0.0
        agent_avgs.append((result.agent_name, avg))
        all_scores.update(agent_scores)

    # Per-agent table
    print(f"\n{'Agent':<25} {'Avg':>5}  Scores")
    print("-" * 70)
    for result in results:
        if result.parse_error:
            print(f"{result.agent_name:<25} {'ERR':>5}  (parse failed)")
            continue
        avg = sum(result.scores.values()) / len(result.scores) if result.scores else 0.0
        scores_str = ", ".join(f"{k}={v:.0f}" for k, v in result.scores.items())
        print(f"{result.agent_name:<25} {avg:>5.1f}  {scores_str}")

    # All scores flat
    if all_scores:
        print(f"\n{'Attribute':<35} {'Score':>5}")
        print("-" * 42)
        for attr, score in all_scores.items():
            bar = "#" * int(score) + "." * (5 - int(score))
            print(f"{attr:<35} {score:>5.0f}  [{bar}]")

        overall_avg = sum(all_scores.values()) / len(all_scores)
        overall_min = min(all_scores.values())
        lowest_attr = min(all_scores, key=all_scores.get)
        print(f"\n  Overall avg:    {overall_avg:.2f}")
        print(f"  Overall min:    {overall_min:.0f} ({lowest_attr})")
        print(f"  Total attrs:    {len(all_scores)}")

    # Repair notes
    has_repairs = any(r.repair and r.repair != "No issues." for r in results if not r.parse_error)
    if has_repairs:
        print(f"\n{'REPAIR NOTES':=^70}")
        for result in results:
            if result.parse_error or not result.repair:
                continue
            if result.repair == "No issues.":
                continue
            print(f"\n  [{result.agent_name}]")
            # Wrap long repair text
            words = result.repair.split()
            line = "    "
            for word in words:
                if len(line) + len(word) + 1 > 70:
                    print(line)
                    line = "    " + word
                else:
                    line += " " + word if line.strip() else "    " + word
            if line.strip():
                print(line)

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Debug VLM judge with multiple specialized agents"
    )
    parser.add_argument("--shoe", required=True, help="Path to reference shoe image")
    parser.add_argument("--candidate", required=True, help="Path to generated/candidate image")
    parser.add_argument(
        "--vlm", choices=list(VLM_MODELS), default="8b",
        help="VLM variant (default: 8b)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=10240,
        help="Max new tokens per agent (default: 10240)",
    )
    args = parser.parse_args()

    import torch
    from PIL import Image as PILImage
    from casadei.media import ImageMedia, TextMedia, MediaBundle
    from casadei.models.registry import default_registry

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    # Load images
    shoe_img = PILImage.open(args.shoe).convert("RGB")
    candidate_img = PILImage.open(args.candidate).convert("RGB")
    print(f"Shoe:      {args.shoe}  ({shoe_img.size[0]}x{shoe_img.size[1]})")
    print(f"Candidate: {args.candidate}  ({candidate_img.size[0]}x{candidate_img.size[1]})")
    print(f"VLM:       {VLM_MODELS[args.vlm]}")
    print(f"Agents:    {', '.join(a.name for a in AGENTS)}")
    print("=" * 70)

    # Load model once
    model_cls = default_registry.get(VLM_MODELS[args.vlm])
    model = model_cls()
    print("Loading model...")
    model.load_model()

    # Override max_new_tokens
    model.DEFAULT_PARAMS = {**model.DEFAULT_PARAMS, "max_new_tokens": args.max_tokens}

    # Run each agent sequentially
    results: list[AgentResult] = []
    for i, agent in enumerate(AGENTS):
        print(f"\n{'─' * 70}")
        print(f"Agent {i + 1}/{len(AGENTS)}: {agent.name}")
        print(f"Features: {', '.join(agent.features)}")
        print(f"{'─' * 70}")

        result = _run_agent(
            model, agent, shoe_img, candidate_img,
            ImageMedia, TextMedia, MediaBundle,
        )
        results.append(result)

    # Print combined results
    _print_results(results)

    # Cleanup
    print("\nUnloading model...")
    model.unload_model()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
