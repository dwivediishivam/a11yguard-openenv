"""Task loader for the A11yGuard environment."""

import json
import base64
import os
import random
from pathlib import Path
from typing import Optional, Dict, Any, List


DATA_DIR = Path(__file__).parent.parent / "data"


def list_difficulties() -> List[str]:
    """Return available difficulty levels."""
    return ["easy", "medium", "hard"]


def list_tasks(difficulty: str) -> List[str]:
    """List all task IDs for a given difficulty."""
    diff_dir = DATA_DIR / difficulty
    if not diff_dir.exists():
        return []
    tasks = sorted([
        d.name for d in diff_dir.iterdir()
        if d.is_dir() and (d / "code.html").exists() and (d / "ground_truth.json").exists()
    ])
    return tasks


def load_task(difficulty: str, task_name: str) -> Dict[str, Any]:
    """Load a specific task by difficulty and name.

    Returns dict with keys: html_code, ground_truth, screenshot_base64, task_id, page_description
    """
    task_dir = DATA_DIR / difficulty / task_name

    # Load HTML code
    html_path = task_dir / "code.html"
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    html_code = html_path.read_text(encoding="utf-8")

    # Load ground truth
    gt_path = task_dir / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Load screenshot if available
    screenshot_base64 = None
    screenshot_path = task_dir / "screenshot.png"
    if screenshot_path.exists():
        with open(screenshot_path, "rb") as f:
            screenshot_base64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "task_id": f"{difficulty}/{task_name}",
        "difficulty": difficulty,
        "html_code": html_code,
        "ground_truth": ground_truth,
        "screenshot_base64": screenshot_base64,
        "page_description": ground_truth.get("page_description", ""),
    }


def load_random_task(difficulty: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Load a random task from the given difficulty level."""
    tasks = list_tasks(difficulty)
    if not tasks:
        raise ValueError(f"No tasks found for difficulty: {difficulty}")

    if seed is not None:
        rng = random.Random(seed)
        task_name = rng.choice(tasks)
    else:
        task_name = random.choice(tasks)

    return load_task(difficulty, task_name)


def get_task_summary() -> Dict[str, Any]:
    """Get a summary of all available tasks."""
    summary = {}
    for diff in list_difficulties():
        tasks = list_tasks(diff)
        summary[diff] = {
            "count": len(tasks),
            "tasks": tasks,
        }
    return summary
