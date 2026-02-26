# Structured Judge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the free-text ACCEPT/REJECT judge with attribute-based scoring per the approved design at `docs/plans/2026-02-26-structured-judge-design.md`.

**Architecture:** Three-file change. `loop.py` exposes iteration context to judges. `judge.py` gets feature extraction + structured scoring with tolerance-based accept/reject. `run_shoe_tryon_loop.py` wires `--tolerance` and `extract_features()` into the pipeline.

**Tech Stack:** Python, PIL, Qwen3-VL-8B (via casadei VLMSession)

---

### Task 1: Add iteration context to LoopStep

**Files:**
- Modify: `src/casadei/loop.py:169-187` (inside the `for i in range(...)` loop)

**Step 1: Write the failing test**

In `tests/test_loop_step.py`, add a test that verifies `loop_iteration` and `loop_max_iterations` are set in context when the judge is called:

```python
class TestLoopStepIterationContext:
    """Test: judge receives loop_iteration and loop_max_iterations in context."""

    def test_iteration_context_passed_to_judge(self):
        agent = _make_mock_agent()
        step = AgentStep(
            name="gen",
            agent=agent,
            input_map={"image": "person"},
            output_map={"image": "image"},
            template_kwargs={"feedback": ""},
        )

        captured = []

        def judge(ctx):
            captured.append({
                "loop_iteration": ctx.get("loop_iteration"),
                "loop_max_iterations": ctx.get("loop_max_iterations"),
            })
            if len(captured) < 2:
                return False, "Try again."
            return True, "OK."

        loop = LoopStep(
            name="test_loop",
            body=[step],
            judge=judge,
            max_iterations=5,
        )

        context = {"person": _make_image()}
        loop.execute(context)

        assert len(captured) == 2
        assert captured[0]["loop_iteration"] == 0
        assert captured[0]["loop_max_iterations"] == 5
        assert captured[1]["loop_iteration"] == 1
        assert captured[1]["loop_max_iterations"] == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_loop_step.py::TestLoopStepIterationContext -v`
Expected: FAIL — `loop_iteration` is `None`

**Step 3: Implement — set iteration context before judge call**

In `src/casadei/loop.py`, inside `execute()`, add two lines before the `self.judge(working)` call (around line 182):

```python
                # Set iteration context for the judge
                working["loop_iteration"] = i
                working["loop_max_iterations"] = self.max_iterations

                # Always judge — we need the verdict and feedback for logging
                print(f"  Judging...")
                accepted, feedback = self.judge(working)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_loop_step.py::TestLoopStepIterationContext -v`
Expected: PASS

**Step 5: Run all existing loop tests**

Run: `pytest tests/test_loop_step.py -v`
Expected: All 11+ tests PASS (no regressions)

---

### Task 2: Add `extract_features()` to judge.py

**Files:**
- Modify: `workflows/shoe_tryon_loop/scripts/judge.py` (add function near top, after helpers)

**Step 1: Implement `extract_features()`**

Add after the `_stream_vlm()` helper (around line 96):

```python
_FEATURE_PROMPT = """\
Look at this shoe image. List 4-6 short visual attribute names that \
describe its key features (e.g. color, heel type, material, toe shape, sole style).

Reply with ONLY a JSON array of short strings. Example:
["red color", "block heel", "platform sole", "pointed toe", "patent leather"]
"""

_FALLBACK_FEATURES = ["color", "shape", "heel", "material", "details"]


def extract_features(session: VLMSession, shoe_image: ImageMedia) -> list[str]:
    """Ask the VLM to identify visual attributes of the reference shoe.

    Called once before the loop starts. Returns a list of short attribute names.
    Falls back to generic features on parse failure.
    """
    import json as _json

    bundle = MediaBundle(items={
        "shoe": shoe_image,
        "prompt": TextMedia(text=_FEATURE_PROMPT),
    })

    model = session.acquire()
    try:
        response = _stream_vlm(model, bundle, label="Feature Extraction")
    finally:
        session.release()

    # Parse JSON array from response
    try:
        # Find the JSON array in the response (may have surrounding text)
        start = response.index("[")
        end = response.index("]", start) + 1
        features = _json.loads(response[start:end])
        if isinstance(features, list) and all(isinstance(f, str) for f in features) and len(features) >= 2:
            print(f"  Extracted features: {features}")
            return features
    except (ValueError, _json.JSONDecodeError):
        pass

    print(f"  Feature extraction parse failed, using fallback: {_FALLBACK_FEATURES}")
    return list(_FALLBACK_FEATURES)
```

**Step 2: Verify syntax**

Run: `python -c "import sys; sys.path.insert(0, 'workflows/shoe_tryon_loop/scripts'); import judge; print('OK')"`
Expected: OK (no import errors)

---

### Task 3: Rewrite `make_judge()` with structured scoring

**Files:**
- Modify: `workflows/shoe_tryon_loop/scripts/judge.py` — replace `_JUDGE_PROMPT`, `_MAX_RETRIES`, and rewrite `make_judge()`

**Step 1: Replace the judge prompt and tolerance config**

Replace `_JUDGE_PROMPT` and `_MAX_RETRIES` with:

```python
_SCORE_PROMPT = """\
You are a quality inspector for a virtual shoe try-on system.

IMAGE 1 is the REFERENCE shoe — the original product photo.
IMAGE 2 is the TRY-ON RESULT — a person who should be wearing \
the exact same shoe from IMAGE 1 on their feet.

Your job: examine the shoes on the person's feet in IMAGE 2 and \
rate how closely they match the reference shoe in IMAGE 1.

{iteration_context}\
{previous_feedback}\
{stale_nudge}\
Score each attribute from 1 to 5:
  1 = completely different from the reference
  2 = vaguely similar but clearly wrong
  3 = recognizably the same shoe type but noticeable differences
  4 = close match with only minor differences
  5 = near-identical to the reference

Attributes to score: {features}

Reply in this exact format:
SCORES: {score_format}
REPAIR: <describe what is wrong in the result, then tell the model \
to look at the reference shoe image and match that part — do NOT \
give direct orders like "make it X", instead point out the mismatch \
and direct the model to use the reference shoe image to fix it>
"""

_MAX_RETRIES = 3

TOLERANCE_CONFIGS = {
    "generous":  {"avg_threshold": 2.5, "min_floor": 1.5},
    "moderate":  {"avg_threshold": 3.5, "min_floor": 2.5},
    "strict":    {"avg_threshold": 4.5, "min_floor": 3.5},
}
```

**Step 2: Add score parsing helper**

```python
def _parse_scores(text: str, features: list[str]) -> dict[str, float]:
    """Parse 'SCORES: attr1=N, attr2=N, ...' from VLM response.

    Returns dict mapping feature name -> score. Raises ValueError on failure.
    """
    # Find SCORES: line
    scores_match = re.search(r"SCORES:\s*(.+)", text, re.IGNORECASE)
    if not scores_match:
        raise ValueError("No 'SCORES:' line found")

    scores_text = scores_match.group(1).strip()
    scores = {}

    for pair in re.split(r",\s*", scores_text):
        pair = pair.strip()
        if not pair:
            continue
        m = re.match(r"(.+?)\s*=\s*([0-9]+(?:\.[0-9]+)?)", pair)
        if m:
            attr_name = m.group(1).strip()
            scores[attr_name] = float(m.group(2))

    if not scores:
        raise ValueError(f"Could not parse any scores from: {scores_text}")

    return scores


def _parse_repair(text: str) -> str:
    """Extract REPAIR text from VLM response."""
    repair_match = re.search(r"REPAIR:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if repair_match:
        return repair_match.group(1).strip()
    return text
```

**Step 3: Rewrite `make_judge()`**

Replace the entire `make_judge()` function with the new structured scoring version:

```python
def make_judge(
    session: VLMSession,
    shoe_key: str = "shoe",
    candidate_key: str = "image",
    features: list[str] | None = None,
    tolerance: str = "generous",
) -> JudgeCallable:
    """Return a structured scoring JudgeCallable.

    Uses per-attribute 1-5 scoring with code-controlled accept/reject.
    Includes stale feedback guardrail and iteration context.
    """
    if features is None:
        features = list(_FALLBACK_FEATURES)

    tol = TOLERANCE_CONFIGS.get(tolerance, TOLERANCE_CONFIGS["generous"])
    avg_threshold = tol["avg_threshold"]
    min_floor = tol["min_floor"]

    # Closure state for stale guardrail
    prev_lowest_attr: list[str | None] = [None]
    stale_count: list[int] = [0]
    prev_feedback: list[str] = [""]

    score_format = ", ".join(f"{f}=N" for f in features)

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        candidate = context.get(candidate_key)
        reference = context.get(shoe_key)

        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."
        if not isinstance(reference, ImageMedia):
            return False, f"Missing reference shoe image (key='{shoe_key}')."

        # Read iteration context from LoopStep
        iteration = context.get("loop_iteration", 0)
        max_iterations = context.get("loop_max_iterations", 5)

        # Build dynamic prompt sections
        iteration_context = f"This is attempt {iteration + 1} of {max_iterations}.\n"

        previous_feedback = ""
        if prev_feedback[0]:
            previous_feedback = f"Previous feedback: \"{prev_feedback[0]}\"\n"

        stale_nudge = ""
        if stale_count[0] >= 2 and prev_lowest_attr[0]:
            attr = prev_lowest_attr[0]
            n = stale_count[0]
            stale_nudge = (
                f"The attribute '{attr}' has been the weakest for {n} consecutive "
                f"attempts. In your REPAIR instruction, clearly describe what is wrong "
                f"with '{attr}' in the generated result, then tell the model to look at "
                f"the reference shoe image and match that specific part. Do NOT give "
                f"direct orders like 'make it X'. Instead, point out the mismatch and "
                f"direct the model to use the reference shoe image as the source of truth.\n"
            )

        prompt_text = _SCORE_PROMPT.format(
            iteration_context=iteration_context,
            previous_feedback=previous_feedback,
            stale_nudge=stale_nudge,
            features=", ".join(features),
            score_format=score_format,
        )

        bundle = MediaBundle(items={
            "reference": reference,
            "candidate": candidate,
            "prompt": TextMedia(text=prompt_text),
        })

        model = session.acquire()
        try:
            last_error = ""
            raw_response = ""

            for attempt in range(_MAX_RETRIES + 1):
                if attempt > 0 and last_error:
                    # Retry with error feedback
                    retry_prompt = (
                        f"{prompt_text}\n\n"
                        f"Your previous response could not be parsed. Error: {last_error}\n"
                        f"Please reply in the exact format:\n"
                        f"SCORES: {score_format}\n"
                        f"REPAIR: ..."
                    )
                    bundle = MediaBundle(items={
                        "reference": reference,
                        "candidate": candidate,
                        "prompt": TextMedia(text=retry_prompt),
                    })

                raw_response = _stream_vlm(model, bundle, label="VLM Judge")

                try:
                    scores = _parse_scores(raw_response, features)
                    repair = _parse_repair(raw_response)

                    # Accept/reject logic
                    avg = sum(scores.values()) / len(scores)
                    lowest_val = min(scores.values())
                    lowest_attr = min(scores, key=scores.get)
                    accepted = avg >= avg_threshold and lowest_val >= min_floor

                    # Update stale guardrail
                    if lowest_attr == prev_lowest_attr[0]:
                        stale_count[0] += 1
                    else:
                        prev_lowest_attr[0] = lowest_attr
                        stale_count[0] = 1

                    # Update previous feedback
                    prev_feedback[0] = repair

                    # Print summary
                    verdict = "ACCEPT" if accepted else "REJECT"
                    scores_str = ", ".join(f"{k}={v}" for k, v in scores.items())
                    print(f"  [{verdict}] avg={avg:.1f} min={lowest_val:.1f} "
                          f"(threshold: avg>={avg_threshold}, min>={min_floor})")
                    print(f"  Scores: {scores_str}")
                    if not accepted:
                        print(f"  Repair: {repair[:200]}")

                    return accepted, repair

                except ValueError as e:
                    last_error = str(e)
                    print(f"  [VLM Judge] Parse error: {last_error} "
                          f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})")

            # All retries exhausted — REJECT with raw text
            print(f"  [VLM Judge] Fallback: treating unparseable response as REJECT")
            fallback = raw_response or "Could not parse VLM response."
            prev_feedback[0] = fallback
            return False, fallback
        finally:
            session.release()

    return judge
```

**Step 4: Verify syntax**

Run: `python -c "import sys; sys.path.insert(0, 'workflows/shoe_tryon_loop/scripts'); import judge; print('OK')"`
Expected: OK

---

### Task 4: Wire `--tolerance` and `extract_features()` into the test runner

**Files:**
- Modify: `tests/run_shoe_tryon_loop.py`

**Step 1: Add `--tolerance` argument**

After the `--scale` argument (line ~240), add:

```python
    parser.add_argument(
        "--tolerance", type=str, default="generous",
        choices=["generous", "moderate", "strict"],
        help="Judge tolerance level (default: generous)",
    )
```

**Step 2: Update `build_pipeline()` to accept features and tolerance**

```python
def build_pipeline(
    max_iterations: int = 5,
    num_inference_steps: int = 30,
    swap_models: bool = True,
    vlm_session: VLMSession | None = None,
    features: list[str] | None = None,
    tolerance: str = "generous",
) -> Pipeline:
```

And update the `make_judge()` call to pass `features` and `tolerance`:

```python
        judge=make_judge(
            session=vlm_session,
            shoe_key="shoe",
            candidate_key="image",
            features=features,
            tolerance=tolerance,
        ),
```

**Step 3: Call `extract_features()` in `main()` and pass through**

Update the import to include `extract_features`:

```python
from judge import VLMSession, make_judge, make_best_fn, extract_features
```

After creating `vlm_session` and before `build_pipeline()`, add feature extraction:

```python
    # Extract shoe features for structured judge
    print("Extracting shoe features...")
    shoe_media = ImageMedia(image=shoes)
    vlm_session_temp_loaded = not args.keep_both
    if vlm_session_temp_loaded:
        vlm_session.acquire()  # temp load for extraction
    else:
        vlm_session.load()  # pre-load for keep-both
    features = extract_features(vlm_session, shoe_media)
    if vlm_session_temp_loaded:
        vlm_session.release()
    print(f"Features: {features}")
    print(f"Tolerance: {args.tolerance}")
    print()
```

Update `build_pipeline()` call:

```python
    pipeline = build_pipeline(
        max_iterations=args.max_iter,
        num_inference_steps=args.steps,
        swap_models=not args.keep_both,
        vlm_session=vlm_session,
        features=features,
        tolerance=args.tolerance,
    )
```

Also print tolerance in the header:

```python
    print(f"Tolerance: {args.tolerance}")
```

**Step 4: Update results saving to include scores**

In `save_results()`, add tolerance to results_data:

```python
    results_data["tolerance"] = args.tolerance if hasattr(args, 'tolerance') else "generous"
```

Actually, tolerance is already implicit from the scores. No change needed — the feedback field already captures the REPAIR text, and scores can be added later if needed.

---

### Task 5: Run all tests and verify

**Step 1: Run unit tests**

Run: `pytest tests/test_loop_step.py -v`
Expected: All tests PASS

**Step 2: Verify judge.py imports cleanly**

Run: `python -c "import sys; sys.path.insert(0, 'workflows/shoe_tryon_loop/scripts'); from judge import extract_features, make_judge, make_best_fn, VLMSession, TOLERANCE_CONFIGS; print('All imports OK')"`
Expected: "All imports OK"

---
