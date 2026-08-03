"""
4-Step Tactical Attack Demo & Calibration on User Full-Base Screenshots
-----------------------------------------------------------------------
Loads user-uploaded screenshots ('demo_fullbase.PNG', 'demo_fullbase1.PNG', 'demo_fullbase2.PNG'),
simulates the 4-Step Tactical Attack Engine, and generates visual annotated attack plan overlays
('debug_attack_plan_demo_fullbase.png') showing:
- Green points: Outermost border Troop drops (Step 1)
- Magenta points: Hero drops after troops (Step 2)
- Yellow/Orange circles: Rage Spells A BIT AHEAD of troops toward core defenses (Step 3)
- Text banner: Hero Ability Activation timeline (Step 4)
"""

import os
import cv2
import numpy as np
from src.agent.predetermined_agent import PredeterminedAttacker


def annotate_and_save_demo(image_path: str, save_path: str, troop_type: str = "VALKYRIE") -> None:
    if not os.path.exists(image_path):
        print(f"[WARN] Demo image '{image_path}' not found. Skipping.")
        return

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w, _ = img.shape
    overlay = img.copy()

    attacker = PredeterminedAttacker(use_fast_pipeline=True)

    # 1. Draw Step 1 (Troops First along outermost green border)
    for side_name, pts in attacker.sides_geometry.items():
        px_pts = [(int(p[0] * w), int(p[1] * h)) for p in pts]
        for i in range(len(px_pts) - 1):
            cv2.line(overlay, px_pts[i], px_pts[i + 1], (0, 255, 0), 2)
        for pt in px_pts:
            cv2.circle(overlay, pt, 6, (0, 255, 0), -1)

    # 2. Draw Step 2 (Heroes after troops - midpoint on each side)
    for side_name, pts in attacker.sides_geometry.items():
        mid = pts[len(pts) // 2]
        px_mid = (int(mid[0] * w), int(mid[1] * h))
        cv2.circle(overlay, px_mid, 12, (255, 0, 255), -1)
        cv2.circle(overlay, px_mid, 14, (255, 255, 255), 2)
        cv2.putText(
            overlay,
            "HERO",
            (px_mid[0] - 25, px_mid[1] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

    # 3. Draw Step 3 (Rage Spells A BIT AHEAD toward center - not at back)
    for side_name, spell_pts in attacker.spell_geometry.items():
        mid_spell = spell_pts[len(spell_pts) // 2]
        px_spell = (int(mid_spell[0] * w), int(mid_spell[1] * h))
        cv2.circle(overlay, px_spell, 45, (0, 215, 255), 3)  # Golden/Yellow Rage ring
        cv2.putText(
            overlay,
            "RAGE (AHEAD)",
            (px_spell[0] - 55, px_spell[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 215, 255),
            2,
        )

        # Draw arrow from outer border toward Rage circle showing inward troop movement
        outer_mid = attacker.sides_geometry[side_name][len(attacker.sides_geometry[side_name]) // 2]
        px_outer = (int(outer_mid[0] * w), int(outer_mid[1] * h))
        cv2.arrowedLine(overlay, px_outer, px_spell, (0, 255, 255), 2, tipLength=0.3)

    # 4. Draw Step 4 Banner (Hero Ability Activation)
    cv2.rectangle(overlay, (20, h - 65), (w - 20, h - 15), (0, 0, 0), -1)
    cv2.rectangle(overlay, (20, h - 65), (w - 20, h - 15), (0, 215, 255), 2)
    cv2.putText(
        overlay,
        "4-STEP TACTICAL PLAN: 1. Troops -> 2. Heroes -> 3. Inward Rage Ahead -> 4. Trigger Hero Abilities @ T=7s",
        (40, h - 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(save_path, overlay)
    print(f"[INFO] Annotated 4-Step Attack Plan saved to '{save_path}'!")


def run_all_demo_calibrations() -> None:
    print("=====================================================================")
    print("      4-STEP TACTICAL ATTACK DEMO ON USER FULL-BASE SCREENSHOTS     ")
    print("=====================================================================\n")

    demos = [
        ("templates/ui/demo_fullbase.PNG", "debug_attack_plan_demo_fullbase.png"),
        ("templates/ui/demo_fullbase1.PNG", "debug_attack_plan_demo_fullbase1.png"),
        ("templates/ui/demo_fullbase2.PNG", "debug_attack_plan_demo_fullbase2.png"),
    ]

    for img_path, save_path in demos:
        annotate_and_save_demo(img_path, save_path)

    print("\n[SUCCESS] Generated visual tactical overlays on all demo base screenshots!")
    print("=====================================================================\n")


if __name__ == "__main__":
    run_all_demo_calibrations()
