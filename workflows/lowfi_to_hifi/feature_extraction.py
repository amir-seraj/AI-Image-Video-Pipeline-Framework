"""Feature extraction with judge loop for lowfi sketches.

Calls Gemini (text-only) to extract design features from a sketch image,
then runs a judge loop (up to 3 iterations) to verify and refine the
description against the actual sketch.
"""
from __future__ import annotations

import logging

from PIL import Image as PILImage

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """\
You are a professional shoe designer analyzing a rough hand-drawn sketch.
Describe every design element you see in precise technical terms. Be exhaustive:

- Shoe type (sandal, bootie, mule, pump, etc.)
- Toe: style (pointed, round, square, almond) and status (open or closed)
- Heel: type (wedge, stiletto, block, flat) and approximate height
- Back: status (open/mule, closed, slingback)
- Upper construction (panels, cutouts, straps, buckles, zippers)
- Ankle details (straps, cuff, collar)
- Sole/platform details
- Any annotations or text visible in the sketch (list them but note they are \
designer notes, not part of the shoe)
- Overall silhouette and proportions

For each part of the shoe (toe, heel, back, ankle, sole), explicitly state \
whether it is open or closed, present or absent.

Be factual. Only describe what you actually see, do not infer or imagine \
elements that are not drawn."""

JUDGE_PROMPT = """\
You are a senior shoe design reviewer. You are given a rough hand-drawn sketch \
and a text description that was extracted from it.

Your job is to carefully compare the text description against the actual sketch \
image and identify:
1. **Errors** — anything described incorrectly (e.g., says "open toe" when the \
toe is clearly closed)
2. **Missing details** — design elements visible in the sketch but not \
mentioned in the description
3. **Hallucinations** — elements described in the text that are not actually \
present in the sketch

For each issue found, explain what is wrong and what the correct observation \
should be.

If the description is fully accurate, say "APPROVED — no issues found."

Here is the description to review:

---
{description}
---

Compare this carefully against the sketch image provided."""

REFINE_PROMPT = """\
You are a professional shoe designer. You previously analyzed a sketch and \
produced a description, but a reviewer found issues.

Here is your previous description:

---
{description}
---

Here is the reviewer's feedback:

---
{feedback}
---

Please produce a corrected and complete description of the sketch, fixing all \
issues raised by the reviewer. Follow the same format as before. For each part \
of the shoe (toe, heel, back, ankle, sole), explicitly state whether it is \
open or closed, present or absent.

Be factual. Only describe what you actually see in the sketch."""

TEXT_MODEL = "gemini-2.5-flash"
MAX_JUDGE_ITERATIONS = 3


def extract_features(
    sketch: PILImage.Image,
    max_iterations: int = MAX_JUDGE_ITERATIONS,
) -> str:
    """Extract design features from a sketch with judge-loop refinement.

    Args:
        sketch: The lowfi sketch image.
        max_iterations: Max judge review rounds (default 3).

    Returns:
        The final verified feature description text.
    """
    if genai is None:
        raise ImportError(
            "google-genai is required. Install: pip install google-genai"
        )

    client = genai.Client()
    img = sketch.copy()
    img.thumbnail((1024, 1024), PILImage.LANCZOS)

    # Initial extraction
    logger.info("Feature extraction: initial pass")
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[EXTRACT_PROMPT, img],
        config=genai_types.GenerateContentConfig(temperature=0.2),
    )
    description = response.text
    logger.debug("Initial description:\n%s", description)

    # Judge loop
    for i in range(max_iterations):
        logger.info("Feature extraction: judge iteration %d/%d", i + 1, max_iterations)

        judge_prompt = JUDGE_PROMPT.format(description=description)
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=[judge_prompt, img],
            config=genai_types.GenerateContentConfig(temperature=0.2),
        )
        feedback = response.text
        logger.debug("Judge feedback:\n%s", feedback)

        if "APPROVED" in feedback.upper() and "NO ISSUES" in feedback.upper():
            logger.info("Feature extraction: approved at iteration %d", i + 1)
            break

        # Refine
        logger.info("Feature extraction: refining based on feedback")
        refine_prompt = REFINE_PROMPT.format(
            description=description, feedback=feedback
        )
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=[refine_prompt, img],
            config=genai_types.GenerateContentConfig(temperature=0.2),
        )
        description = response.text
        logger.debug("Refined description:\n%s", description)

    return description
