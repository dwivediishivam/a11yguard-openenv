"""Main A11yGuard OpenEnv environment implementation."""

from typing import Tuple, Optional, Dict, Any
from env.models import (
    AccessibilityObservation,
    AccessibilityAction,
    AccessibilityReward,
    EnvironmentState,
)
from env.grader import grade_audit
from env.tasks import load_random_task, load_task


class AccessibilityEnv:
    """OpenEnv environment for web accessibility auditing.

    The agent receives HTML source code (and optionally a screenshot) of a webpage,
    and must identify WCAG 2.1 AA accessibility violations, locate them in the code,
    and suggest fixes.
    """

    def __init__(self, difficulty: str = "easy", task_name: Optional[str] = None, seed: Optional[int] = None):
        """Initialize the environment.

        Args:
            difficulty: One of 'easy', 'medium', 'hard'
            task_name: Specific task to load (e.g., 'task_01'). Random if None.
            seed: Random seed for task selection.
        """
        self.difficulty = difficulty
        self.task_name = task_name
        self.seed = seed
        self.max_steps = 1  # Single-step audit: agent submits one report

        # Internal state
        self._task_data: Optional[Dict[str, Any]] = None
        self._step_number: int = 0
        self._done: bool = False
        self._last_reward: Optional[float] = None

    def reset(self) -> AccessibilityObservation:
        """Reset the environment with a new (or specific) task.

        Returns the initial observation containing the HTML code and screenshot.
        """
        if self.task_name:
            self._task_data = load_task(self.difficulty, self.task_name)
        else:
            self._task_data = load_random_task(self.difficulty, self.seed)

        self._step_number = 0
        self._done = False
        self._last_reward = None

        # Build observation
        gt = self._task_data["ground_truth"]
        hint = gt.get("total_violations", None) if self.difficulty == "easy" else None

        return AccessibilityObservation(
            task_id=self._task_data["task_id"],
            difficulty=self._task_data["difficulty"],
            html_code=self._task_data["html_code"],
            screenshot_base64=self._task_data.get("screenshot_base64"),
            wcag_version="WCAG 2.1 AA",
            total_violations_hint=hint,
            step_number=self._step_number,
            prior_feedback=None,
            page_description=self._task_data.get("page_description"),
        )

    def step(self, action: AccessibilityAction) -> Tuple[AccessibilityObservation, float, bool, Dict[str, Any]]:
        """Process the agent's accessibility audit report.

        Args:
            action: The agent's audit containing detected violations.

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._task_data is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_number += 1
        gt = self._task_data["ground_truth"]

        # Grade the audit
        reward_breakdown = grade_audit(action.violations, gt)
        reward = reward_breakdown.total_reward
        self._last_reward = reward

        # Episode ends after one step (single-submission audit)
        self._done = True

        # Build feedback observation
        obs = AccessibilityObservation(
            task_id=self._task_data["task_id"],
            difficulty=self._task_data["difficulty"],
            html_code=self._task_data["html_code"],
            screenshot_base64=self._task_data.get("screenshot_base64"),
            wcag_version="WCAG 2.1 AA",
            total_violations_hint=None,
            step_number=self._step_number,
            prior_feedback=f"Audit graded. Detection: {reward_breakdown.detection_score:.2f}, "
                          f"Location: {reward_breakdown.location_score:.2f}, "
                          f"Fix quality: {reward_breakdown.fix_score:.2f}. "
                          f"Total reward: {reward:.2f}",
            page_description=self._task_data.get("page_description"),
        )

        info = {
            "reward_breakdown": reward_breakdown.model_dump(),
            "ground_truth_violations": gt.get("total_violations", len(gt.get("violations", []))),
            "reported_violations": len(action.violations),
        }

        return obs, reward, self._done, info

    def state(self) -> EnvironmentState:
        """Return the current internal state of the environment."""
        if self._task_data is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return EnvironmentState(
            task_id=self._task_data["task_id"],
            difficulty=self._task_data["difficulty"],
            step_number=self._step_number,
            done=self._done,
            last_reward=self._last_reward,
            ground_truth_violations=self._task_data["ground_truth"].get(
                "total_violations",
                len(self._task_data["ground_truth"].get("violations", [])),
            ),
            html_code=self._task_data["html_code"],
            screenshot_available=self._task_data.get("screenshot_base64") is not None,
        )

    def close(self):
        """Clean up resources."""
        self._task_data = None
        self._done = True
