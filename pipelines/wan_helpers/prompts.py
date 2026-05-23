"""Articulated-object I2V prompt builder for Wan2.2 (pipeline_v3 Section 5.2 / 5.6).

The user describes ONLY the moving part and its motion direction
(e.g. "the drawer slowly slides outward"). This module appends a universal
camera-lock addon to the positive prompt and constructs a negative prompt
that lists camera-motion artifacts WITHOUT the tokens "static" / "motionless"
(Wan's default sample_neg_prompt contains those, which would actively push
the camera to move and break our locked-off camera assumption).

Templates are loaded from the prompt_templates/ directory as JSON data so
that the Python source itself remains ASCII-only (project code convention).
"""

from __future__ import annotations

import json
import os
from typing import Tuple

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_templates")
_SUPPORTED_LANGS = ("zh", "en")


def _load_template(lang: str) -> dict:
    if lang not in _SUPPORTED_LANGS:
        raise ValueError(f"Unsupported lang: {lang!r}; expected one of {_SUPPORTED_LANGS}")
    path = os.path.join(_TEMPLATE_DIR, f"{lang}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("camera_lock_addon", "neg_prompt"):
        if key not in data or not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"prompt template {path!r} missing or empty key {key!r}")
    return data


def build_articulated_prompts(
    user_object_motion_prompt: str,
    lang: str = "zh",
) -> Tuple[str, str]:
    """Build (pos_prompt, neg_prompt) for Wan2.2 I2V articulated-object generation.

    Parameters
    ----------
    user_object_motion_prompt : str
        Per-object description of the part and its motion direction.
    lang : {"zh", "en"}
        Language of the camera-lock addon and negative prompt.

    Returns
    -------
    pos_prompt : str
        user_object_motion_prompt + universal camera-lock addon (lang-specific).
    neg_prompt : str
        Universal negative prompt enumerating camera-motion artifacts and
        visual quality issues. Deliberately OMITS "static" / "motionless"
        (Wan's default sample_neg_prompt contains them and would penalise
        camera lock).
    """
    if not isinstance(user_object_motion_prompt, str) or not user_object_motion_prompt.strip():
        raise ValueError("user_object_motion_prompt must be a non-empty string")

    template = _load_template(lang)
    pos_prompt = user_object_motion_prompt.strip() + template["camera_lock_addon"]
    neg_prompt = template["neg_prompt"]
    return pos_prompt, neg_prompt
