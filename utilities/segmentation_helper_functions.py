"""
Segmentation helpers for playbook/LLM enrichment: type coercion, prompt building, and JSON parse/validate.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class SegmentationHelper:
    """
    Helper class for segmentation: _num, _int, _str, build_prompt, parse_and_validate.
    Pass task_template (and optionally system_prompt) and allowed taxonomy for prompts/validation.
    """

    DEFAULT_ALLOWED = [
        ("High Deals + Multi-user", "Pipeline-driven", "Forecasting + automation"),
        ("High Email", "Outreach-focused", "Sequences + tracking"),
        ("High Contacts only", "Early stage", "Pipeline setup"),
        ("Single-user heavy", "Expansion opportunity", "Team invites"),
        ("Multi-user low activity", "Stalled", "Reactivation"),
        ("Balanced usage", "High intent", "Direct upgrade"),
    ]
    DEFAULT_ALLOWED_URGENCY = {"reach_out_now", "nurture", "reactivate"}

    def __init__(
        self,
        task_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        allowed: Optional[List[Tuple[str, str, str]]] = None,
        allowed_urgency: Optional[set] = None,
        prompt_task_path: Optional[str] = None,
        prompt_system_path: Optional[str] = None,
    ):
        if task_template is not None:
            self.task_template = task_template
        elif prompt_task_path is not None:
            path = Path(prompt_task_path)
            self.task_template = path.read_text(encoding="utf-8")
        else:
            self.task_template = ""

        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif prompt_system_path is not None:
            path = Path(prompt_system_path)
            self.system_prompt = path.read_text(encoding="utf-8").strip()
        else:
            self.system_prompt = ""

        allowed = allowed or self.DEFAULT_ALLOWED
        self.allowed_behavior = {x[0] for x in allowed}
        self.allowed_stage = {x[1] for x in allowed}
        self.allowed_focus = {x[2] for x in allowed}
        self.allowed_urgency = allowed_urgency or self.DEFAULT_ALLOWED_URGENCY

    def _num(self, x: Any, default: float = 0.0) -> float:
        """Convert input to float; return default for None/NaN or on error."""
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return default
            return float(x)
        except Exception:
            return default

    def _int(self, x: Any, default: int = 0) -> int:
        """Convert input to int (via _num); return default on error."""
        return int(round(self._num(x, default)))

    def _str(self, x: Any, default: str = "") -> str:
        """Convert input to string; return default for None/NaN."""
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return str(x)

    def build_prompt(self, row: pd.Series) -> str:
        """Fill the task prompt with 30-day signals from the given row."""
        return self.task_template.format(
            ID=self._str(row.get("id", row.get("id"))),
            INDUSTRY=self._str(row.get("INDUSTRY")),
            EMPLOYEE_RANGE=self._str(row.get("EMPLOYEE_RANGE")),
            ALEXA_RANK=self._str(row.get("ALEXA_RANK")),
            P=self._num(row.get("P_CONVERT_30D", row.get("P", 0.0)), 0.0),
            DEALS_30=self._int(row.get("DEALS_30", row.get("ACTIONS_CRM_DEALS_30D_SUM", 0))),
            EMAIL_30=self._int(row.get("EMAIL_30", row.get("ACTIONS_EMAIL_30D_SUM", 0))),
            UDEALS_30=self._int(row.get("UDEALS_30", row.get("USERS_CRM_DEALS_30D_SUM", 0))),
            UEMAIL_30=self._int(row.get("UEMAIL_30", row.get("USERS_EMAIL_30D_SUM", 0))),
        )

    def parse_and_validate(self, text: str) -> Dict[str, Any]:
        """
        Parse LLM JSON output and validate category values against allowed taxonomy.
        Strips markdown code fences (e.g. ```json ... ```) before parsing.
        Returns a dict with behavior_pattern, likely_stage, playbook_focus, urgency, subject_line, opening_line, genai_raw.
        """
        empty = {
            "Behaviour Pattern": None,
            "Likely Stage": None,
            "Playbook Focus": None,
            "Urgency": None,
            "Subject Line": None,
            "Opening Line": None,
        }

        text = (text or "").strip()
        if text.startswith("```"):
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        try:
            obj = json.loads(text)
        except Exception:
            return empty

        behavior = obj.get("behavior_pattern")
        stage = obj.get("likely_stage")
        focus = obj.get("playbook_focus")

        if behavior not in self.allowed_behavior:
            behavior = None
        if stage not in self.allowed_stage:
            stage = None
        if focus not in self.allowed_focus:
            focus = None

        urgency = obj.get("urgency")
        if urgency not in self.allowed_urgency:
            urgency = None

        subject = obj.get("subject_line")
        opening = obj.get("opening_line")

        return {
            "Behaviour Pattern": behavior,
            "Likely Stage": stage,
            "Playbook Focus": focus,
            "Urgency": urgency,
            "Subject Line": subject,
            "Opening Line": opening,
        }
