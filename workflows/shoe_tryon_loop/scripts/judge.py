"""Judge and best-selection functions for the shoe try-on loop.

Provides factory functions that return callables matching the
JudgeCallable and BestFn protocols defined in casadei.loop.
"""

from __future__ import annotations

import gc
import re
import sys

from PIL import Image as PILImage, ImageDraw, ImageFont

from casadei.loop import BestFn, JudgeCallable, LoopIteration
from casadei.media import ImageMedia, Media, TextMedia, MediaBundle

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating a virtual shoe try-on result.

IMAGE 1 is the REFERENCE shoe — the target design you must compare against.
IMAGE 2 is the CANDIDATE try-on result on a person.

Compare the candidate shoes to the REFERENCE shoe in IMAGE 1. Check: color, \
material/texture, heel shape and height, toe shape, sole style, and overall silhouette.

Answer with exactly one word on the first line: ACCEPT or REJECT.

If REJECT, write a REPAIR INSTRUCTION for an image-editing model. The instruction \
must always refer back to the reference shoe image. Do NOT use absolute descriptions \
like "make the color red". Instead, describe how the candidate differs FROM THE \
REFERENCE and instruct the model to match the reference.

Example repair instructions:
- "The shoes are darker than the reference shoe image. Match the bright red color of the reference shoe image."
- "The heel is too thin compared to the reference shoe image. Match the chunky block heel shown in the reference shoe image."
- "The material looks like velvet but the reference shoe image shows patent leather. Match the glossy patent finish of the reference shoe image."
- "The toe is too rounded. Match the pointed toe shape shown in the reference shoe image."

Always say "the reference shoe image" so the editing model knows to look at it.
Do NOT write a critique. Write what needs to change to match the reference shoe image.
"""

_BEST_PROMPT = """\
You are selecting the best virtual shoe try-on result.

IMAGE 1 is the REFERENCE shoe — the target design.
IMAGE 2 shows {n} candidate try-on results side by side, labeled Option 1 through \
Option {n} from left to right.

Which option's shoes best match the reference in style, color, heel shape, \
material, and natural placement on the person?

Reply with ONLY a single number (1 to {n}) on the first line.
Then on the next line, briefly explain why.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_RETRIES = 2  # extra attempts on unparseable VLM output


def _unload_and_cleanup(model) -> None:
    """Unload a model and clear GPU memory."""
    model.unload_model()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _stream_vlm(model, bundle: MediaBundle, label: str = "VLM") -> str:
    """Run VLM with streaming, printing tokens to terminal in real time.

    Returns the full collected response text.
    """
    sys.stdout.write(f"\n  [{label}] ")
    sys.stdout.flush()
    chunks: list[str] = []
    for chunk in model.run_streaming(bundle):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        chunks.append(chunk)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(chunks).strip()


def _concat_candidates(images: list[PILImage.Image], label_height: int = 40) -> PILImage.Image:
    """Stitch candidate images side by side with 'Option N' labels on top."""
    if not images:
        raise ValueError("No images to concat")

    # Resize all to same height for a clean grid
    target_h = max(img.height for img in images)
    resized = []
    for img in images:
        if img.height != target_h:
            ratio = target_h / img.height
            new_w = int(img.width * ratio)
            resized.append(img.resize((new_w, target_h), PILImage.LANCZOS))
        else:
            resized.append(img)

    total_w = sum(img.width for img in resized)
    canvas = PILImage.new("RGB", (total_w, target_h + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except (OSError, IOError):
        font = ImageFont.load_default()

    x_offset = 0
    for i, img in enumerate(resized):
        label = f"Option {i + 1}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x_offset + (img.width - text_w) // 2
        draw.text((text_x, 6), label, fill=(0, 0, 0), font=font)
        canvas.paste(img, (x_offset, label_height))
        x_offset += img.width

    return canvas


# ---------------------------------------------------------------------------
# VLM Session
# ---------------------------------------------------------------------------

class VLMSession:
    """Shared VLM model lifecycle for judge and best_fn.

    In swap mode (default), the model is loaded and unloaded on each
    acquire/release pair. In keep-loaded mode (call ``load()`` first),
    the model stays resident and ``release()`` is a no-op.
    """

    def __init__(self, model_name: str = "qwen3_vl_8b"):
        self._model_name = model_name
        self._model = None
        self._persistent = False

    def load(self) -> None:
        """Pre-load the model (keep-both mode). Call once before the loop."""
        if self._model is None:
            from casadei.models.registry import default_registry
            model_cls = default_registry.get(self._model_name)
            self._model = model_cls()
            self._model.load_model()
        self._persistent = True
        print(f"  [VLMSession] Pre-loaded {self._model_name} (keep-both mode)")

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self._model is not None:
            _unload_and_cleanup(self._model)
            self._model = None
        self._persistent = False

    def acquire(self):
        """Get the model, loading it if not already loaded."""
        if self._model is None:
            from casadei.models.registry import default_registry
            model_cls = default_registry.get(self._model_name)
            self._model = model_cls()
            self._model.load_model()
        return self._model

    def release(self) -> None:
        """Release the model. Unloads only if not in persistent mode."""
        if not self._persistent and self._model is not None:
            _unload_and_cleanup(self._model)
            self._model = None


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def make_judge(
    session: VLMSession,
    shoe_key: str = "shoe",
    candidate_key: str = "image",
) -> JudgeCallable:
    """Return a JudgeCallable using a shared VLMSession.

    Streams the VLM response to stdout in real time. Retries on
    unparseable output (no ACCEPT/REJECT on first line).
    """

    def _run_once(model, bundle: MediaBundle) -> tuple[bool, str, str]:
        """Single judge attempt. Returns (accepted, feedback, raw_response)."""
        response_text = _stream_vlm(model, bundle, label="VLM Judge")

        first_line = response_text.splitlines()[0].strip().upper() if response_text else ""

        if "ACCEPT" in first_line:
            accepted = True
        elif "REJECT" in first_line:
            accepted = False
        else:
            # Unparseable — signal caller to retry
            return False, "", ""

        lines = response_text.splitlines()
        feedback = " ".join(lines[1:]).strip() if len(lines) > 1 else response_text
        return accepted, feedback, response_text

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        candidate = context.get(candidate_key)
        reference = context.get(shoe_key)

        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}') in context."
        if not isinstance(reference, ImageMedia):
            return False, f"Missing reference shoe image (key='{shoe_key}') in context."

        bundle = MediaBundle(items={
            "reference": reference,
            "candidate": candidate,
            "prompt": TextMedia(text=_JUDGE_PROMPT),
        })

        model = session.acquire()
        try:
            # Try up to _MAX_RETRIES + 1 times on unparseable output
            for attempt in range(_MAX_RETRIES + 1):
                accepted, feedback, raw = _run_once(model, bundle)
                if raw:  # got a parseable response
                    verdict = "ACCEPT" if accepted else "REJECT"
                    print(f"  [{verdict}]")
                    if not accepted:
                        print(f"  Repair: {feedback[:200]}")
                    return accepted, feedback

                # Unparseable — retry
                print(f"  [VLM Judge] Could not parse ACCEPT/REJECT (attempt {attempt + 1}/{_MAX_RETRIES + 1}), retrying...")

            # All retries exhausted — treat as REJECT with raw text as feedback
            print(f"  [VLM Judge] Fallback: treating unparseable response as REJECT")
            # Use the last raw response we got (from _stream_vlm)
            fallback_feedback = "Could not parse VLM response. Please try again."
            return False, fallback_feedback
        finally:
            session.release()

    return judge


# ---------------------------------------------------------------------------
# Best-of-N Selection
# ---------------------------------------------------------------------------

def make_best_fn(
    session: VLMSession,
    shoe_key: str = "shoe",
    output_key: str = "image",
) -> BestFn:
    """Return a BestFn that concatenates all candidates into a single labeled
    image and asks the VLM to pick the best option.

    Sends exactly 2 images to the VLM: the reference shoe + a grid of all
    candidates with "Option N" labels. Streams the response to stdout.
    Retries on unparseable output; falls back to last candidate.
    """

    def best_fn(
        history: list[LoopIteration],
        context: dict[str, Media],
    ) -> dict[str, Media]:
        candidates: list[ImageMedia] = []
        for record in history:
            img = record.outputs.get(output_key)
            if isinstance(img, ImageMedia):
                candidates.append(img)

        if not candidates:
            return {}
        if len(candidates) == 1:
            print(f"\n  [VLM Best-of-N] Only 1 candidate, selecting it.")
            return {
                output_key: candidates[0],
                "best_selection_index": 1,
                "best_selection_reason": TextMedia(text="Only one candidate."),
            }

        reference = context.get(shoe_key)
        if not isinstance(reference, ImageMedia):
            return {output_key: candidates[-1]}

        # Concat all candidates into a single labeled image
        n = len(candidates)
        pil_images = [c.image for c in candidates]
        grid_image = _concat_candidates(pil_images)

        print(f"\n  [VLM Best-of-N] Selecting from {n} candidates (grid: {grid_image.size[0]}x{grid_image.size[1]})")

        prompt_text = _BEST_PROMPT.format(n=n)
        bundle = MediaBundle(items={
            "reference": reference,
            "candidates_grid": ImageMedia(image=grid_image),
            "prompt": TextMedia(text=prompt_text),
        })

        model = session.acquire()
        try:
            chosen_1based = None
            response = ""

            for attempt in range(_MAX_RETRIES + 1):
                response = _stream_vlm(model, bundle, label="VLM Best-of-N")

                # Parse: find a number 1..n on the first line
                first_line = response.splitlines()[0] if response else ""
                match = re.search(r"\d+", first_line)
                if match:
                    val = int(match.group())
                    if 1 <= val <= n:
                        chosen_1based = val
                        break

                print(f"  [VLM Best-of-N] Could not parse valid option 1-{n} (attempt {attempt + 1}/{_MAX_RETRIES + 1}), retrying...")

            # Fallback: pick last candidate (most refined)
            if chosen_1based is None:
                chosen_1based = n
                print(f"  [VLM Best-of-N] Fallback: selecting last candidate (Option {n})")
        finally:
            session.release()

        chosen_0based = chosen_1based - 1
        print(f"  Selected: Option {chosen_1based} of {n}")

        return {
            output_key: candidates[chosen_0based],
            "best_selection_index": chosen_1based,
            "best_selection_reason": TextMedia(text=response),
        }

    return best_fn
