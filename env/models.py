"""Pydantic models for the A11yGuard OpenEnv environment."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ViolationReport(BaseModel):
    """A single accessibility violation reported by the agent."""
    violation_type: str = Field(description="Snake-case violation type, e.g. 'missing_alt_text'")
    wcag_criterion: str = Field(description="WCAG criterion ID, e.g. '1.1.1'")
    line_numbers: List[int] = Field(description="Line numbers in the HTML where the violation occurs")
    description: str = Field(description="Human-readable description of the violation")
    suggested_fix: str = Field(description="Corrected HTML snippet or description of the fix")


class AccessibilityAction(BaseModel):
    """Action submitted by the agent: a full accessibility audit report."""
    violations: List[ViolationReport] = Field(default_factory=list, description="List of detected violations")
    overall_verdict: Literal["accessible", "partially_accessible", "inaccessible"] = Field(
        default="partially_accessible",
        description="Overall accessibility verdict"
    )


class AccessibilityObservation(BaseModel):
    """Observation returned to the agent each step."""
    task_id: str = Field(description="Unique task identifier, e.g. 'easy/task_01'")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Task difficulty level")
    html_code: str = Field(description="Full HTML source code to audit")
    screenshot_base64: Optional[str] = Field(default=None, description="Base64-encoded PNG screenshot of the page")
    wcag_version: str = Field(default="WCAG 2.1 AA", description="WCAG standard to audit against")
    total_violations_hint: Optional[int] = Field(default=None, description="Hint: expected number of violations (for easy tasks only)")
    step_number: int = Field(default=0, description="Current step number in the episode")
    prior_feedback: Optional[str] = Field(default=None, description="Feedback from previous step, if any")
    page_description: Optional[str] = Field(default=None, description="Brief description of the page being audited")


class AccessibilityReward(BaseModel):
    """Reward breakdown returned after each step."""
    detection_score: float = Field(description="Fraction of actual violations correctly identified")
    location_score: float = Field(description="Fraction of detected violations with correct line numbers")
    fix_score: float = Field(description="Fraction of detected violations with valid fixes")
    total_reward: float = Field(description="Weighted total reward strictly in (0, 1)")
    details: dict = Field(default_factory=dict, description="Per-violation grading details")

    @field_validator("detection_score", "location_score", "fix_score", "total_reward", mode="before")
    @classmethod
    def clamp_scores(cls, v):
        """Ensure all scores are strictly between 0 and 1."""
        if isinstance(v, (int, float)):
            return max(0.01, min(0.99, float(v)))
        return v


class EnvironmentState(BaseModel):
    """Full internal state of the environment."""
    task_id: str
    difficulty: str
    step_number: int
    done: bool
    last_reward: Optional[float] = None
    ground_truth_violations: int
    html_code: str
    screenshot_available: bool
