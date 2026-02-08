"""Verify that FSLSM dimension vectors shift after a profile update.

Run from the repo root:
    python backend/tests/test_fslsm_update.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base.llm_factory import LLMFactory
from modules.adaptive_learner_modeling.agents.adaptive_learning_profiler import (
    update_learner_profile_with_llm,
)

BEFORE_PROFILE = {
    "learner_information": "MBA grad with admin background",
    "learning_goal": "Become an HR Manager",
    "cognitive_status": {
        "overall_progress": 20,
        "mastered_skills": [
            {"name": "Communication", "proficiency_level": "advanced"}
        ],
        "in_progress_skills": [
            {
                "name": "HRIS Management",
                "required_proficiency_level": "intermediate",
                "current_proficiency_level": "unlearned",
            }
        ],
    },
    "learning_preferences": {
        "fslsm_dimensions": {
            "fslsm_processing": 0.0,
            "fslsm_perception": 0.0,
            "fslsm_input": 0.0,
            "fslsm_understanding": 0.0,
        }
    },
    "behavioral_patterns": {
        "system_usage_frequency": "2 logins/week",
        "session_duration_engagement": "20 min avg",
    },
}

# Interaction that strongly signals: active, visual, sequential
INTERACTIONS = {
    "feedback": (
        "I loved the hands-on exercises and video walkthroughs. "
        "Step-by-step labs are way more effective for me than reading theory."
    )
}


def main():
    llm = LLMFactory.create(model="gpt-4o", model_provider="openai")

    print("Sending profile update to LLM...")
    after = update_learner_profile_with_llm(
        llm, BEFORE_PROFILE, INTERACTIONS, "", None
    )

    dims_before = BEFORE_PROFILE["learning_preferences"]["fslsm_dimensions"]
    dims_after = after["learning_preferences"]["fslsm_dimensions"]

    print()
    print("Dimension          Before  →  After")
    print("─" * 42)
    any_changed = False
    for key in dims_before:
        label = key.replace("fslsm_", "")
        b = dims_before[key]
        a = dims_after[key]
        changed = b != a
        any_changed = any_changed or changed
        marker = " *" if changed else ""
        print(f"{label:<18} {b:+.2f}   →  {a:+.2f}{marker}")

    print()
    if any_changed:
        print("PASS — at least one FSLSM dimension changed after the interaction.")
    else:
        print("FAIL — no FSLSM dimensions changed. The LLM may not be following the prompt.")

    # Sanity-check: values must be in [-1, 1]
    for key, val in dims_after.items():
        assert -1 <= val <= 1, f"{key} out of range: {val}"


if __name__ == "__main__":
    main()
