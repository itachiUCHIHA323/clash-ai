"""
UI Matcher & Auto-Navigation Module (Lobby & Battle Buttons)
------------------------------------------------------------
Uses the user-provided UI templates (attack.png, find.PNG, next.PNG,
attack_final.PNG, surrender.PNG) to detect and tap UI buttons automatically.
"""

import os
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List


class UIMatcher:
    """
    Detects and interacts with Clash of Clans UI buttons using OpenCV Template Matching.
    Searches both the root directory and 'templates/ui/'.
    """

    def __init__(self, template_paths: Optional[List[str]] = None):
        self.search_dirs = template_paths or [".", "templates/ui"]
        self.templates: Dict[str, np.ndarray] = {}
        self._load_ui_templates()

    def _load_ui_templates(self) -> None:
        """Load UI button images (attack.png, find.PNG, confirm_attack.png, etc.)."""
        # Canonical mappings from filenames to button types
        filename_map = {
            "attack": "attack",
            "find": "find",
            "find_match": "find",
            "attack_final": "attack_final",
            "confirm_attack": "attack_final",
            "next": "next",
            "return": "return",
            "return_home": "return",
            "surrender": "surrender",
            "sur_end": "surrender",
            "end_battle": "surrender",
            "okay_sur": "dialog_ok",
            "okay_back": "dialog_ok",
            "okay_bonus": "dialog_ok",
        }

        for search_dir in self.search_dirs:
            if not os.path.exists(search_dir):
                continue
            for filename in os.listdir(search_dir):
                name_lower = os.path.splitext(filename)[0].lower()
                if name_lower in filename_map:
                    canonical_name = filename_map[name_lower]
                    if canonical_name not in self.templates:
                        filepath = os.path.join(search_dir, filename)
                        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                        if img is not None:
                            self.templates[canonical_name] = img
                            print(f"[INFO] Loaded UI button template '{canonical_name}' from '{filepath}' ({img.shape[1]}x{img.shape[0]})")

    def find_button(self, frame: np.ndarray, button_name: str, threshold: float = 0.75) -> Optional[Tuple[int, int]]:
        """
        Locate a UI button on the screen using template matching.

        :param frame: Full screen screenshot (BGR).
        :param button_name: Name of button ('attack', 'find', 'next', 'attack_final', 'surrender').
        :param threshold: Matching confidence threshold (0.0 to 1.0).
        :return: (X, Y) center coordinate of the button if found, otherwise None.
        """
        button_name = button_name.lower()
        if button_name not in self.templates:
            return None

        tmpl = self.templates[button_name]
        th, tw, _ = tmpl.shape
        h, w, _ = frame.shape
        if tw > w or th > h:
            return None

        res = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            match_x = max_loc[0] + tw // 2
            match_y = max_loc[1] + th // 2
            return (int(match_x), int(match_y))
        return None

    def is_visible(self, frame: np.ndarray, button_name: str, threshold: float = 0.75) -> bool:
        """Return True if the specified button is currently visible on screen."""
        return self.find_button(frame, button_name, threshold=threshold) is not None


if __name__ == "__main__":
    matcher = UIMatcher()
    print("Loaded UI templates:", list(matcher.templates.keys()))
