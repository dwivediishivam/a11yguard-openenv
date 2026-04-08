# A11yGuard: Web Accessibility Audit Environment

**An OpenEnv environment that evaluates whether AI agents can audit web pages for WCAG 2.1 AA accessibility compliance.**

---

## The Problem

In 2025, AI agents and code generators produce an ever-growing share of the web. GitHub Copilot, Cursor, v0, and countless AI tools write HTML, CSS, and full web applications daily. But **accessibility is almost never part of the equation**.

The numbers are stark:
- **96.3%** of the top 1 million websites have detectable WCAG failures ([WebAIM Million, 2024](https://webaim.org/projects/million/))
- The **EU Web Accessibility Directive** and the **US Americans with Disabilities Act (ADA)** legally require accessible websites
- Over **1 billion people worldwide** live with some form of disability

The result: AI is producing millions of inaccessible web pages, creating barriers for users who rely on screen readers, keyboard navigation, or other assistive technologies.

**A11yGuard** exists to answer a critical question: _Can AI agents identify and fix the accessibility problems that AI itself is creating?_

---

## What Makes A11yGuard Unique

### Dual-Modality Auditing
Unlike rule-based tools (axe, Lighthouse) that only analyze the DOM, A11yGuard gives agents both:
- **Source code** (HTML/CSS) — for structural analysis
- **Visual screenshot** — for catching issues like low contrast, visual-only information, and layout problems

This mirrors how human accessibility experts actually work: reading the code _and_ looking at the page.

### Real-World Web Pages
Every task is a complete, realistic webpage — not a contrived snippet. You'll audit:
- E-commerce stores, dashboards, media players
- Booking forms, admin panels, social feeds
- Course platforms, HR portals, property listings

### Grading Beyond Detection
The grader evaluates three dimensions:
1. **Detection** (50%): Did the agent find the real violations?
2. **Location** (30%): Did it point to the correct line numbers?
3. **Fix Quality** (20%): Did it suggest valid fixes?

This rewards agents that don't just flag problems but can _pinpoint and fix them_.

---

## Environment Overview

| Property | Value |
|---|---|
| **Standard** | WCAG 2.1 Level AA |
| **Tasks** | 3 (easy, medium, hard) |
| **Examples** | 36 total (12 per difficulty) |
| **API** | `reset()` → `step()` → `state()` |
| **Reward Range** | 0.0 – 1.0 |
| **Episode Length** | 1 step (single audit submission) |
| **Input** | HTML source code + optional PNG screenshot |
| **Output** | Structured violation report with line numbers and fixes |

---

## Task Descriptions

### Easy Audit
**1-2 obvious violations per page.** Single, clear WCAG failures that any accessibility-aware developer would catch.

Examples: missing alt text, missing lang attribute, no page title, empty buttons, focus outline removed, non-descriptive link text.

Expected baseline score: **0.5 – 0.8**

### Medium Audit
**2-3 moderate violations per page.** Subtler issues requiring understanding of ARIA patterns, semantic HTML, and form accessibility.

Examples: tables without headers, radio groups without fieldset/legend, modals missing dialog role, accordions without aria-expanded, placeholders used as labels.

Expected baseline score: **0.3 – 0.6**

### Hard Audit
**5-7 violations per page across multiple categories.** Complex real-world pages with interrelated accessibility failures spanning navigation, widgets, forms, and content.

Examples: e-commerce pages with hover-only interactions + emoji buttons + missing labels; dashboards with inaccessible charts + low contrast + non-semantic navigation; media players with unlabeled controls + custom sliders.

Expected baseline score: **0.1 – 0.4**

---

## Action & Observation Spaces

### Observation (what the agent receives)
```python
{
    "task_id": "easy/task_01",
    "difficulty": "easy",
    "html_code": "<!DOCTYPE html>...",        # Full HTML source
    "screenshot_base64": "iVBORw0KGgo...",    # Base64 PNG (optional)
    "wcag_version": "WCAG 2.1 AA",
    "total_violations_hint": 2,               # Only for easy tasks
    "step_number": 0,
    "prior_feedback": null,
    "page_description": "A tech blog article..."
}
```

### Action (what the agent submits)
```python
{
    "violations": [
        {
            "violation_type": "missing_alt_text",
            "wcag_criterion": "1.1.1",
            "line_numbers": [28, 34],
            "description": "Images missing alt attribute",
            "suggested_fix": "<img src=\"logo.png\" alt=\"Company logo\">"
        }
    ],
    "overall_verdict": "inaccessible"
}
```

### Reward (0.0 – 1.0)
```
total = 0.5 × detection + 0.3 × location + 0.2 × fix_quality
```
- **detection**: fraction of real violations correctly identified (fuzzy type matching)
- **location**: fraction of found violations with correct line numbers (±3 line tolerance)
- **fix_quality**: fraction of suggested fixes that contain the key remediation elements

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)
- HuggingFace account with API token

### Local Development
```bash
# Clone and install
git clone https://huggingface.co/spaces/dwivediishivam/openenv_scaler_hack_web-accessibility-auditor
cd a11yguard
pip install -r requirements.txt

# Run the server
python server/main.py
# Server starts at http://localhost:7860

# Run baseline inference
export HF_TOKEN=your_token_here
python inference.py
```

### Docker
```bash
docker build -t a11yguard .
docker run -p 7860:7860 a11yguard
```

### API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode. Body: `{"difficulty": "easy"}` |
| `/step` | POST | Submit audit. Body: `{"action": {...}}` |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List all available tasks |

---

## Baseline Performance

Model: `google/gemma-4-31B-it:novita` via HuggingFace Inference Router

| Task | Score | Detection | Location | Fix Quality |
|---|---|---|---|---|
| Easy | **0.97** | 1.00 | 1.00 | 0.83 |
| Medium | **0.82** | 1.00 | 0.67 | 0.50 |
| Hard | **0.35** | 0.40 | 0.30 | 0.20 |
| **Average** | **0.71** | | | |

_Baseline run with seed=42. Hard tasks show significant difficulty progression as expected._

---

## WCAG Violations Covered

A11yGuard covers violations across all four WCAG principles:

**Perceivable**
- 1.1.1 Non-text Content (alt text)
- 1.3.1 Info and Relationships (semantic HTML, labels, headings)
- 1.3.5 Identify Input Purpose (autocomplete)
- 1.4.1 Use of Color
- 1.4.2 Audio Control
- 1.4.3 Contrast (Minimum)

**Operable**
- 2.1.1 Keyboard
- 2.2.2 Pause, Stop, Hide
- 2.4.1 Bypass Blocks (skip nav)
- 2.4.2 Page Titled
- 2.4.4 Link Purpose
- 2.4.7 Focus Visible

**Understandable**
- 3.1.1 Language of Page
- 3.2.5 Change on Request
- 3.3.1 Error Identification

**Robust**
- 4.1.2 Name, Role, Value
- 4.1.3 Status Messages

---

## Project Structure
```
a11yguard/
├── inference.py          # Baseline inference script
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile            # Container config
├── requirements.txt      # Python dependencies
├── env/
│   ├── models.py         # Pydantic models
│   ├── grader.py         # Grading logic
│   ├── tasks.py          # Task loader
│   └── accessibility_env.py  # Main environment
├── server/
│   └── main.py           # FastAPI server
└── data/
    ├── easy/             # 12 easy tasks
    ├── medium/           # 12 medium tasks
    └── hard/             # 12 hard tasks
```

---

## License

MIT

---

_Built for the Meta × Scaler OpenEnv Hackathon 2025._
_A11yGuard: Because the web should be accessible to everyone — and AI should help make it so._
