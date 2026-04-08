"""Grading logic for the A11yGuard accessibility audit environment."""

import re
from typing import List, Dict, Any
from env.models import ViolationReport, AccessibilityReward


# Mapping of violation types to broader categories for fuzzy matching
VIOLATION_CATEGORIES = {
    "missing_alt_text": ["alt", "image", "img", "non-text"],
    "missing_lang_attribute": ["lang", "language"],
    "missing_page_title": ["title", "page title"],
    "non_descriptive_link_text": ["click here", "link text", "link purpose", "non-descriptive"],
    "missing_form_labels": ["label", "form label", "input label"],
    "missing_form_label": ["label", "form label", "input label"],
    "missing_input_label": ["label", "form label", "input label"],
    "empty_button_no_accessible_name": ["button", "accessible name", "aria-label", "empty button"],
    "heading_level_skip": ["heading", "h1", "h2", "h3", "h4", "heading hierarchy", "heading skip"],
    "autoplay_video_no_controls": ["autoplay", "video", "audio", "media", "controls"],
    "focus_outline_removed": ["focus", "outline", "focus indicator", "focus visible"],
    "focus_indicator_removed": ["focus", "outline", "focus indicator", "focus visible"],
    "empty_link": ["empty link", "link text", "anchor"],
    "color_only_indication": ["color only", "use of color", "color alone"],
    "interactive_div_not_button": ["div", "button", "semantic", "interactive", "keyboard"],
    "icon_button_no_accessible_name": ["icon", "button", "aria-label", "accessible name"],
    "insufficient_color_contrast": ["contrast", "color contrast"],
    "no_skip_navigation": ["skip", "skip nav", "bypass"],
    "non_semantic_navigation": ["nav", "navigation", "semantic", "landmark"],
    "table_missing_headers": ["table", "header", "th", "scope"],
    "table_headers_using_td": ["table", "header", "th", "td"],
    "table_missing_caption": ["table", "caption"],
    "radio_group_no_fieldset": ["fieldset", "legend", "radio", "group"],
    "radio_missing_labels": ["radio", "label"],
    "checkbox_missing_labels": ["checkbox", "label"],
    "modal_missing_role": ["modal", "dialog", "role"],
    "modal_missing_aria_modal": ["modal", "aria-modal"],
    "modal_missing_aria_labelledby": ["modal", "aria-labelledby", "labelledby"],
    "placeholder_as_label": ["placeholder", "label"],
    "link_opens_new_tab_no_warning": ["new tab", "target blank", "target", "window"],
    "error_not_linked_to_field": ["error", "aria-describedby", "error message"],
    "error_not_associated_with_field": ["error", "aria-describedby", "error message"],
    "icon_links_no_accessible_name": ["icon", "link", "aria-label", "accessible name"],
    "tooltip_not_keyboard_accessible": ["tooltip", "keyboard"],
    "auto_scrolling_no_pause": ["scroll", "pause", "marquee", "ticker", "animation"],
    "no_aria_live_for_ticker": ["aria-live", "live region", "ticker"],
    "accordion_no_aria_expanded": ["accordion", "aria-expanded", "expanded"],
    "accordion_uses_div_not_button": ["accordion", "div", "button"],
    "accordion_no_aria_controls": ["accordion", "aria-controls"],
    "ambiguous_button_text": ["ambiguous", "button text", "identical", "duplicate"],
    "checkbox_label_not_associated": ["checkbox", "label"],
    "autocomplete_disabled_on_personal_fields": ["autocomplete", "personal"],
    "progress_indicator_not_accessible": ["progress", "stepper", "indicator"],
    "nav_uses_div_not_semantic": ["nav", "div", "semantic", "navigation"],
    "hover_only_quick_view": ["hover", "keyboard", "mouse only"],
    "color_swatch_no_accessible_name": ["color", "swatch", "accessible name"],
    "newsletter_input_no_label": ["newsletter", "label", "input"],
    "sidebar_links_no_accessible_name": ["sidebar", "link", "accessible name"],
    "chart_no_text_alternative": ["chart", "graph", "text alternative", "visualization"],
    "alert_not_announced": ["alert", "aria-live", "announcement"],
    "sidebar_not_nav_landmark": ["sidebar", "nav", "landmark", "navigation"],
    "stepper_no_aria": ["stepper", "step", "aria", "progress"],
    "filter_tags_not_buttons": ["filter", "tag", "button", "div"],
    "quantity_buttons_no_labels": ["quantity", "button", "label", "plus", "minus"],
    "duplicate_add_buttons": ["duplicate", "add", "button", "identical"],
    "allergen_warning_no_role": ["allergen", "warning", "alert", "role"],
    "missing_lang": ["lang", "language"],
    "media_controls_no_labels": ["media", "control", "button", "label"],
    "slider_no_aria": ["slider", "range", "aria-value", "progress"],
    "transcript_toggle_not_button": ["transcript", "toggle", "button"],
    "search_button_no_label": ["search", "button", "label"],
    "view_toggle_not_buttons": ["toggle", "view", "button"],
    "contact_links_no_accessible_name": ["contact", "link", "accessible name"],
    "search_input_no_label": ["search", "input", "label"],
    "pagination_no_nav_landmark": ["pagination", "nav", "landmark"],
    "video_controls_no_labels": ["video", "control", "button", "label"],
    "progress_bar_no_aria": ["progress", "slider", "aria-value"],
    "action_buttons_are_divs": ["action", "button", "div"],
    "tabs_no_aria_pattern": ["tab", "tablist", "aria-selected"],
    "module_accordion_no_semantics": ["module", "accordion", "button", "expanded"],
    "lesson_items_not_interactive": ["lesson", "interactive", "keyboard", "link"],
    "gallery_images_no_alt": ["gallery", "image", "alt"],
    "booking_inputs_no_labels": ["booking", "input", "label"],
    "guest_select_no_label": ["guest", "select", "label"],
    "map_no_text_alternative": ["map", "text alternative"],
    "star_rating_no_text_alternative": ["star", "rating", "text alternative"],
    "image_missing_alt": ["image", "alt", "img"],
}


def _normalize_type(violation_type: str) -> str:
    """Normalize a violation type string for matching."""
    return re.sub(r'[^a-z0-9]', '_', violation_type.lower().strip()).strip('_')


def _match_violation(reported: ViolationReport, gt_violation: dict) -> float:
    """Score how well a reported violation matches a ground truth violation.
    Returns a score between 0.0 and 1.0.
    """
    gt_type = _normalize_type(gt_violation["violation_type"])
    reported_type = _normalize_type(reported.violation_type)

    # Exact type match
    if gt_type == reported_type:
        return 1.0

    # Check if reported type contains the ground truth type or vice versa
    if gt_type in reported_type or reported_type in gt_type:
        return 0.9

    # Keyword-based fuzzy match
    gt_keywords = set()
    if gt_type in VIOLATION_CATEGORIES:
        gt_keywords = set(VIOLATION_CATEGORIES[gt_type])

    reported_keywords = set()
    if reported_type in VIOLATION_CATEGORIES:
        reported_keywords = set(VIOLATION_CATEGORIES[reported_type])

    # Also extract keywords from the description
    reported_desc_lower = reported.description.lower()
    gt_desc_lower = gt_violation.get("description", "").lower()

    # Check WCAG criterion match
    wcag_match = reported.wcag_criterion.strip() == gt_violation.get("wcag_criterion", "").strip()

    # Check keyword overlap from categories
    if gt_keywords and reported_keywords:
        overlap = len(gt_keywords & reported_keywords)
        if overlap >= 2:
            return 0.8

    # Check if the WCAG criterion matches and descriptions share key terms
    if wcag_match:
        gt_terms = set(gt_desc_lower.split())
        reported_terms = set(reported_desc_lower.split())
        overlap = len(gt_terms & reported_terms)
        if overlap >= 5:
            return 0.7

    # Last resort: check description similarity
    gt_terms = set(re.findall(r'\b\w+\b', gt_desc_lower))
    reported_terms = set(re.findall(r'\b\w+\b', reported_desc_lower))
    common_meaningful = gt_terms & reported_terms - {'the', 'a', 'an', 'is', 'are', 'for', 'to', 'of', 'in', 'no', 'not', 'with', 'has', 'have', 'that', 'this', 'and', 'or'}
    if len(common_meaningful) >= 6:
        return 0.6

    return 0.0


def _check_line_numbers(reported: ViolationReport, gt_violation: dict, tolerance: int = 3) -> float:
    """Check if reported line numbers are close to ground truth. Returns 0.0 or 1.0."""
    gt_lines = set(gt_violation.get("line_numbers", []))
    reported_lines = set(reported.line_numbers)

    if not gt_lines or not reported_lines:
        return 0.0

    matches = 0
    for r_line in reported_lines:
        for gt_line in gt_lines:
            if abs(r_line - gt_line) <= tolerance:
                matches += 1
                break

    return min(1.0, matches / len(gt_lines))


def _check_fix_quality(reported: ViolationReport, gt_violation: dict) -> float:
    """Check if the suggested fix addresses the violation. Returns 0.0-1.0."""
    fix = reported.suggested_fix.lower().strip()
    ref_fix = gt_violation.get("reference_fix", "").lower().strip()
    vtype = gt_violation["violation_type"]

    if not fix:
        return 0.0

    score = 0.0

    # Check if fix contains key elements that address the violation
    fix_checks = {
        "missing_alt_text": ["alt=", "alt ="],
        "missing_lang_attribute": ["lang="],
        "missing_page_title": ["<title"],
        "non_descriptive_link_text": ["<a"],
        "missing_form_labels": ["<label", "aria-label"],
        "missing_form_label": ["<label", "aria-label"],
        "missing_input_label": ["<label", "aria-label"],
        "empty_button_no_accessible_name": ["aria-label"],
        "heading_level_skip": ["<h2", "<h3"],
        "autoplay_video_no_controls": ["controls"],
        "focus_outline_removed": ["outline", "focus"],
        "focus_indicator_removed": ["outline", "focus"],
        "empty_link": ["<a", "text"],
        "color_only_indication": ["aria-", "required", "label"],
        "interactive_div_not_button": ["<button"],
        "icon_button_no_accessible_name": ["aria-label"],
        "insufficient_color_contrast": ["color:", "#"],
        "no_skip_navigation": ["skip", "main"],
        "non_semantic_navigation": ["<nav"],
        "table_missing_headers": ["<th", "scope"],
        "table_headers_using_td": ["<th", "scope"],
        "table_missing_caption": ["<caption"],
        "radio_group_no_fieldset": ["<fieldset", "<legend"],
        "modal_missing_role": ["role=", "dialog"],
        "placeholder_as_label": ["<label"],
        "link_opens_new_tab_no_warning": ["new tab", "opens"],
        "error_not_linked_to_field": ["aria-describedby"],
        "error_not_associated_with_field": ["aria-describedby"],
        "accordion_no_aria_expanded": ["aria-expanded"],
        "accordion_uses_div_not_button": ["<button"],
        "tabs_no_aria_pattern": ["role=", "tab"],
        "slider_no_aria": ["role=", "slider", "aria-value"],
        "autocomplete_disabled_on_personal_fields": ["autocomplete="],
    }

    check_keywords = fix_checks.get(_normalize_type(vtype), [])

    if check_keywords:
        matches = sum(1 for kw in check_keywords if kw in fix)
        score = min(1.0, matches / max(1, len(check_keywords) * 0.5))
    else:
        # Fallback: check overlap with reference fix
        if ref_fix:
            ref_terms = set(re.findall(r'\b\w+\b', ref_fix))
            fix_terms = set(re.findall(r'\b\w+\b', fix))
            common = ref_terms & fix_terms - {'the', 'a', 'an', 'class', 'div', 'span'}
            if len(common) >= 3:
                score = 0.7
            elif len(common) >= 1:
                score = 0.4

    return score


def grade_audit(
    reported_violations: List[ViolationReport],
    ground_truth: dict,
    detection_weight: float = 0.5,
    location_weight: float = 0.3,
    fix_weight: float = 0.2,
) -> AccessibilityReward:
    """Grade an agent's accessibility audit against ground truth.

    Returns an AccessibilityReward with detection, location, fix, and total scores.
    """
    gt_violations = ground_truth.get("violations", [])

    if not gt_violations:
        # No violations expected (clean page)
        if not reported_violations:
            return AccessibilityReward(
                detection_score=1.0,
                location_score=1.0,
                fix_score=1.0,
                total_reward=1.0,
                details={"message": "Correctly identified no violations"},
            )
        else:
            penalty = min(1.0, len(reported_violations) * 0.2)
            return AccessibilityReward(
                detection_score=1.0 - penalty,
                location_score=0.0,
                fix_score=0.0,
                total_reward=max(0.0, 1.0 - penalty),
                details={"message": "False positives reported", "false_positives": len(reported_violations)},
            )

    # Match reported violations to ground truth using best-match assignment
    details: Dict[str, Any] = {"matches": [], "missed": [], "false_positives": []}
    gt_matched = [False] * len(gt_violations)
    reported_matched = [False] * len(reported_violations)

    # Build score matrix
    scores = []
    for i, gt_v in enumerate(gt_violations):
        for j, reported_v in enumerate(reported_violations):
            match_score = _match_violation(reported_v, gt_v)
            if match_score >= 0.6:
                scores.append((match_score, i, j))

    # Greedy best-match assignment
    scores.sort(key=lambda x: -x[0])
    detection_hits = 0
    location_total = 0.0
    fix_total = 0.0

    for match_score, gt_idx, rep_idx in scores:
        if gt_matched[gt_idx] or reported_matched[rep_idx]:
            continue

        gt_matched[gt_idx] = True
        reported_matched[rep_idx] = True
        detection_hits += 1

        loc_score = _check_line_numbers(reported_violations[rep_idx], gt_violations[gt_idx])
        fix_score = _check_fix_quality(reported_violations[rep_idx], gt_violations[gt_idx])

        location_total += loc_score
        fix_total += fix_score

        details["matches"].append({
            "ground_truth": gt_violations[gt_idx]["violation_type"],
            "reported": reported_violations[rep_idx].violation_type,
            "match_confidence": round(match_score, 2),
            "location_correct": round(loc_score, 2),
            "fix_quality": round(fix_score, 2),
        })

    # Record missed and false positives
    for i, matched in enumerate(gt_matched):
        if not matched:
            details["missed"].append(gt_violations[i]["violation_type"])

    for j, matched in enumerate(reported_matched):
        if not matched:
            details["false_positives"].append(reported_violations[j].violation_type)

    # Compute final scores
    n_gt = len(gt_violations)
    detection_score = detection_hits / n_gt if n_gt > 0 else 0.0
    location_score = location_total / detection_hits if detection_hits > 0 else 0.0
    fix_score_avg = fix_total / detection_hits if detection_hits > 0 else 0.0

    # Apply false positive penalty (minor)
    fp_count = len(details["false_positives"])
    fp_penalty = min(0.15, fp_count * 0.03)
    detection_score = max(0.0, detection_score - fp_penalty)

    total_reward = (
        detection_weight * detection_score
        + location_weight * location_score
        + fix_weight * fix_score_avg
    )

    return AccessibilityReward(
        detection_score=round(detection_score, 4),
        location_score=round(location_score, 4),
        fix_score=round(fix_score_avg, 4),
        total_reward=round(total_reward, 4),
        details=details,
    )
