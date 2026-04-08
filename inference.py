"""
A11yGuard Inference Script
===========================
Baseline inference script for evaluating a vision-capable LLM on the
A11yGuard web accessibility audit environment.

Required environment variables:
    API_BASE_URL   - API endpoint for the LLM (default: HuggingFace router)
    MODEL_NAME     - Model identifier (default: google/gemma-4-31B-it:novita)
    HF_TOKEN       - HuggingFace API token (required)

Output format: [START]/[STEP]/[END] as required by OpenEnv hackathon spec.
"""

import json
import os
import sys
import traceback

from openai import OpenAI

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.accessibility_env import AccessibilityEnv
from env.models import AccessibilityAction, ViolationReport

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it:novita")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "a11yguard"
MAX_STEPS = 1  # Single-step audit per episode
TEMPERATURE = 0.3
MAX_TOKENS = 4096

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ── Prompt Construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert web accessibility auditor. You analyze HTML/CSS code for WCAG 2.1 AA compliance.

Given HTML source code (and optionally a screenshot), identify ALL accessibility violations.

For each violation found, provide:
1. violation_type: snake_case name (e.g., "missing_alt_text", "insufficient_color_contrast")
2. wcag_criterion: The specific WCAG criterion (e.g., "1.1.1", "2.4.7")
3. line_numbers: List of line numbers where the violation occurs in the HTML
4. description: Clear explanation of why this is a violation
5. suggested_fix: The corrected HTML code snippet

Respond ONLY with valid JSON in this exact format:
{
  "violations": [
    {
      "violation_type": "...",
      "wcag_criterion": "...",
      "line_numbers": [...],
      "description": "...",
      "suggested_fix": "..."
    }
  ],
  "overall_verdict": "accessible|partially_accessible|inaccessible"
}

Be thorough but precise. Only report real WCAG 2.1 AA violations. Do not report best practices or recommendations that are not actual WCAG failures."""


def build_user_prompt(html_code: str, difficulty: str, page_description: str = "") -> str:
    """Build the user prompt for the LLM."""
    hint = ""
    if difficulty == "easy":
        hint = "\n\nNote: This page has a small number of obvious accessibility violations."
    elif difficulty == "medium":
        hint = "\n\nNote: This page has several moderate accessibility issues. Look for ARIA pattern violations, form markup issues, and semantic HTML problems."
    elif difficulty == "hard":
        hint = "\n\nNote: This is a complex page with multiple accessibility violations across different categories. Be thorough in your analysis."

    desc = f"\nPage description: {page_description}" if page_description else ""

    return f"""Audit the following HTML page for WCAG 2.1 AA accessibility violations.{desc}{hint}

```html
{html_code}
```

Analyze the code carefully and respond with the JSON format specified. Include line numbers relative to the HTML above (first line is line 1)."""


def build_messages(html_code: str, difficulty: str, screenshot_base64: str = None, page_description: str = ""):
    """Build the message list for the API call.

    Always sends text (HTML code). Sends screenshot only if USE_VISION=true is set
    in environment variables AND a screenshot is available.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    text_prompt = build_user_prompt(html_code, difficulty, page_description)

    use_vision = os.getenv("USE_VISION", "false").lower() == "true"

    if use_vision and screenshot_base64:
        # Multimodal: text + image
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}},
            ],
        })
    else:
        # Text-only: safe for any model
        messages.append({"role": "user", "content": text_prompt})

    return messages


# ── Response Parsing ──────────────────────────────────────────────────────────

def parse_llm_response(response_text: str) -> AccessibilityAction:
    """Parse the LLM's JSON response into an AccessibilityAction."""
    # Try to extract JSON from the response
    text = response_text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return AccessibilityAction(violations=[], overall_verdict="partially_accessible")
        else:
            return AccessibilityAction(violations=[], overall_verdict="partially_accessible")

    violations = []
    for v in data.get("violations", []):
        try:
            violations.append(ViolationReport(
                violation_type=v.get("violation_type", "unknown"),
                wcag_criterion=v.get("wcag_criterion", ""),
                line_numbers=v.get("line_numbers", []),
                description=v.get("description", ""),
                suggested_fix=v.get("suggested_fix", ""),
            ))
        except Exception:
            continue

    verdict = data.get("overall_verdict", "partially_accessible")
    if verdict not in ("accessible", "partially_accessible", "inaccessible"):
        verdict = "partially_accessible"

    return AccessibilityAction(violations=violations, overall_verdict=verdict)


# ── Main Inference Loop ───────────────────────────────────────────────────────

def run_task(difficulty: str, seed: int = 42) -> dict:
    """Run inference on a single task and return results."""
    task_name = f"{difficulty}-audit"
    env = AccessibilityEnv(difficulty=difficulty, seed=seed)

    rewards = []
    last_error = None
    steps = 0
    success = False
    score = 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    try:
        obs = env.reset()

        # Build messages — text-only by default, vision only if USE_VISION=true
        messages = build_messages(
            html_code=obs.html_code,
            difficulty=obs.difficulty,
            screenshot_base64=obs.screenshot_base64,
            page_description=obs.page_description or "",
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=120,
        )

        response_text = response.choices[0].message.content or ""

        # Guard: if model returned an HTML error page instead of JSON
        if response_text.strip().startswith("<!DOCTYPE") or response_text.strip().startswith("<html"):
            raise ValueError("Model returned an HTML error page instead of a JSON audit response")

        # Parse response into action
        action = parse_llm_response(response_text)
        action_str = f"audit(violations={len(action.violations)},verdict={action.overall_verdict})"

        # Submit to environment
        obs, reward, done, info = env.step(action)
        steps += 1
        reward = max(0.01, min(0.99, reward))
        rewards.append(reward)
        score = reward

        last_error = None
        success = reward >= 0.1

        print(f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

    except Exception as e:
        steps += 1
        last_error = str(e).replace('\n', ' ')[:200]
        rewards.append(0.01)
        print(f"[STEP] step={steps} action=error reward=0.01 done=true error={last_error}")

    finally:
        env.close()

    # Failsafe: clamp final score to open interval (0, 1)
    score = max(0.01, min(0.99, score))
    rewards = [max(0.01, min(0.99, r)) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")

    return {
        "task": task_name,
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
    }


def main():
    """Run inference across all three difficulty levels."""
    results = []

    for difficulty in ["easy", "medium", "hard"]:
        result = run_task(difficulty, seed=42)
        results.append(result)
        print()  # Blank line between tasks

    # Print summary
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"# Baseline Summary: avg_score={avg_score:.2f}")
    for r in results:
        print(f"#   {r['task']}: score={r['score']:.2f} success={r['success']}")


if __name__ == "__main__":
    main()
