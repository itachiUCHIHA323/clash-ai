"""
Predetermined Attacking Agent (Sneaky Goblins, Valkyries, Dragons, E-Drags & 4 Heroes)
-------------------------------------------------------------------------------------
Automates specialized deployment strategies for specific troop compositions:
1. Surround Deployment (Valkyries / Sneaky Goblins):
   - Deploys troops on ALL 4 sides of the enemy base perimeter.
   - Deploys 4 Heroes (King, Queen, Warden, Champion) with ONE hero on each side.
2. Line Sweep Deployment (Dragons / Electro Dragons):
   - Deploys all Dragons / E-Drags along ANY SINGLE side of the base.
   - Deploys all 4 Heroes alongside the dragons on that exact same side.
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from src.controller.fast_controller import FastController
from src.controller.adb_controller import ADBController
from src.vision.card_scanner import CardScanner
from src.vision.fast_ocr import FastUIReader


class PredeterminedAttacker:
    """
    Executes predetermined troop and hero deployments based on army type and base geometry.
    """

    def __init__(self, use_fast_pipeline: bool = True, device_serial: Optional[str] = None, adb_path: Optional[str] = None):
        self.use_fast_pipeline = use_fast_pipeline
        if self.use_fast_pipeline:
            self.controller = FastController()
            self.ui_reader = FastUIReader()
        else:
            self.controller = ADBController(device_serial=device_serial, adb_path=adb_path)
            self.ui_reader = FastUIReader()

        self.card_scanner = CardScanner()

        # 4-Sided Base Perimeter Geometry (Normalized Coordinates 0.0 - 1.0)
        # Positioned strictly on the OUTERMOST EDGE of the green map border to avoid "cannot deploy here" red zone errors!
        self.sides_geometry = {
            "TOP_LEFT": [
                (0.15, 0.45), (0.20, 0.39), (0.25, 0.33), (0.31, 0.27),
                (0.36, 0.21), (0.42, 0.15)
            ],
            "TOP_RIGHT": [
                (0.58, 0.15), (0.63, 0.21), (0.69, 0.27), (0.74, 0.33),
                (0.80, 0.39), (0.85, 0.45)
            ],
            "BOTTOM_RIGHT": [
                (0.85, 0.55), (0.80, 0.61), (0.74, 0.67), (0.69, 0.73),
                (0.63, 0.79), (0.58, 0.85)
            ],
            "BOTTOM_LEFT": [
                (0.42, 0.85), (0.36, 0.79), (0.31, 0.73), (0.25, 0.67),
                (0.20, 0.61), (0.15, 0.55)
            ],
        }

    def get_screen_dimensions(self) -> Tuple[int, int]:
        """Return width and height of the active screen/window."""
        if self.use_fast_pipeline:
            return self.controller.region["width"], self.controller.region["height"]
        else:
            return self.controller.get_screen_resolution()

    def tap_screen(self, x: int, y: int) -> None:
        """Send a tap command to screen coordinates."""
        if self.use_fast_pipeline:
            self.controller.tap_fast(x, y)
        else:
            self.controller.tap(x, y)
        time.sleep(0.04)

    def tap_normalized(self, norm_x: float, norm_y: float) -> None:
        """Tap at normalized (0.0 to 1.0) screen coordinate."""
        w, h = self.get_screen_dimensions()
        self.tap_screen(int(norm_x * w), int(norm_y * h))

    def select_card_by_name(self, card_name: str, card_map: Dict[str, Tuple[int, int]]) -> bool:
        """
        Select a troop or hero card from the scanned card coordinates map.
        """
        if card_name not in card_map:
            print(f"[WARN] Card '{card_name}' not found in deployment bar map.")
            return False
        cx, cy = card_map[card_name]
        self.tap_screen(cx, cy)
        time.sleep(0.06)
        return True

    def scan_battle_cards(self) -> Dict[str, Tuple[int, int]]:
        """
        Capture a frame and scan the bottom deployment bar for card coordinates.
        """
        frame = (
            self.controller.get_screenshot_fast()
            if self.use_fast_pipeline
            else self.controller.get_screenshot()
        )
        return self.card_scanner.scan_cards(frame)

    def auto_attack(self, troop_type: str = "EDRAGON", side: str = "BOTTOM_LEFT", monitor_duration_sec: int = 20) -> Dict[str, Any]:
        """
        Automatically detect troop type and execute either Surround Deployment (Valkyries/Sneaky Goblins)
        or Line Sweep Deployment (Dragons/E-Drags).

        :param troop_type: 'SNEAKY_GOBLIN', 'VALKYRIE', 'DRAGON', 'EDRAGON'.
        :param side: Which side to attack from for Line Sweep ('TOP_LEFT', 'TOP_RIGHT', 'BOTTOM_RIGHT', 'BOTTOM_LEFT').
        :param monitor_duration_sec: Number of seconds to monitor the attack after deployment.
        """
        troop_type = troop_type.upper()
        print(f"\n=======================================================")
        print(f"   Predetermined Attack Engine - Troop: {troop_type}")
        print(f"=======================================================")

        frame = (
            self.controller.get_screenshot_fast()
            if self.use_fast_pipeline
            else self.controller.get_screenshot()
        )
        army_map = self.card_scanner.scan_available_army(frame)
        print(f"[ARMY DISCOVERY] Live Available Army & Real Counts:")
        for name, info in army_map.items():
            print(f"  -> Card: {name:<14} | Count: {info['count']:<3} | Coordinates: {info['pos']}")

        if troop_type in ["SNEAKY_GOBLIN", "VALKYRIE"]:
            self.execute_surround_attack(troop_type, army_map)
        elif troop_type in ["DRAGON", "EDRAGON"]:
            self.execute_line_sweep_attack(troop_type, side, army_map)
        else:
            raise ValueError(f"Unsupported troop_type: {troop_type}")

        print(f"[INFO] Deployment complete! Monitoring battle for {monitor_duration_sec}s...")
        return self.monitor_battle(monitor_duration_sec)

    def zoom_out_base(self) -> None:
        """Perform pinch-out/zoom-out motion so the entire base is visible before deploying."""
        if hasattr(self.controller, "zoom_out"):
            self.controller.zoom_out()
        else:
            print("[ZOOM OUT] Active controller does not support zoom_out(). Skipping.")

    def execute_surround_attack(self, troop_type: str, army_map: Dict[str, Dict[str, Any]]) -> None:
        """
        Surround Deployment (Valkyries / Sneaky Goblins):
        1. Zoom out base view using pinch-out motion.
        2. Read real available count from army_map. Deploy troops evenly across ALL 4 SIDES along the outermost edge.
        3. Deploy available Heroes (King, Queen, Warden, Champion) — ONE on EACH of the 4 sides.
        """
        print(f"[SURROUND ATTACK] Zooming out base & deploying '{troop_type}' on ALL 4 SIDES along outermost edge...")
        self.zoom_out_base()

        # Step 1: Deploy troops across all 4 sides along the outermost green edge
        if troop_type in army_map:
            cx, cy = army_map[troop_type]["pos"]
            real_count = army_map[troop_type]["count"]
            print(f"  -> Selecting '{troop_type}' at { (cx, cy) } (Real remaining count: {real_count})")
            self.tap_screen(cx, cy)
            time.sleep(0.06)

            # Divide real_count across 4 sides (at least 2 passes per side)
            taps_per_side = max(4, (real_count // 4) + 1)
            for side_name, points in self.sides_geometry.items():
                print(f"  -> Deploying {taps_per_side} units of {troop_type} along outermost edge of side: {side_name}")
                for i in range(taps_per_side):
                    pt = points[i % len(points)]
                    self.tap_normalized(pt[0], pt[1])
                    time.sleep(0.04)

        time.sleep(0.5)

        # Step 2: Deploy 1 Hero on EACH side (only heroes present in army_map)
        print("[SURROUND ATTACK] Deploying available Heroes — ONE on EACH of the 4 sides...")
        hero_side_mapping = {
            "KING": "TOP_LEFT",
            "QUEEN": "TOP_RIGHT",
            "WARDEN": "BOTTOM_RIGHT",
            "CHAMPION": "BOTTOM_LEFT",
        }

        for hero_name, assigned_side in hero_side_mapping.items():
            if hero_name in army_map:
                cx, cy = army_map[hero_name]["pos"]
                print(f"  -> Selecting Hero '{hero_name}' at {(cx, cy)}")
                self.tap_screen(cx, cy)
                time.sleep(0.06)
                points = self.sides_geometry[assigned_side]
                midpoint = points[len(points) // 2]
                print(f"     Deploying Hero '{hero_name}' on {assigned_side} at {midpoint}")
                self.tap_normalized(midpoint[0], midpoint[1])
                time.sleep(0.2)

    def execute_line_sweep_attack(self, troop_type: str, side: str, army_map: Dict[str, Dict[str, Any]]) -> None:
        """
        Line Sweep Deployment (Dragons / Electro Dragons):
        1. Zoom out base view using pinch-out motion.
        2. Read real available count from army_map. Deploy all Dragons / E-Drags along ONE selected side.
        3. Deploy ALL available Heroes alongside the dragons on that EXACT same side.
        """
        side = side.upper()
        if side not in self.sides_geometry:
            raise ValueError(f"Invalid side '{side}'. Must be one of {list(self.sides_geometry.keys())}")

        print(f"[LINE SWEEP ATTACK] Zooming out base & deploying ALL '{troop_type}' along outermost edge of side: {side}...")
        self.zoom_out_base()
        points = self.sides_geometry[side]

        # Step 1: Sweep all Dragons / E-Drags along the selected side
        if troop_type in army_map:
            cx, cy = army_map[troop_type]["pos"]
            real_count = army_map[troop_type]["count"]
            print(f"  -> Selecting '{troop_type}' at {(cx, cy)} (Real remaining count: {real_count})")
            self.tap_screen(cx, cy)
            time.sleep(0.06)

            for i in range(max(6, real_count)):
                pt = points[i % len(points)]
                self.tap_normalized(pt[0], pt[1])
                time.sleep(0.06)

        time.sleep(0.5)

        # Step 2: Deploy all available Heroes alongside the dragons on the SAME side
        print(f"[LINE SWEEP ATTACK] Deploying ALL available HEROES alongside {troop_type} on side: {side}...")
        heroes = ["KING", "QUEEN", "WARDEN", "CHAMPION"]
        for i, hero_name in enumerate(heroes):
            if hero_name in army_map:
                cx, cy = army_map[hero_name]["pos"]
                print(f"  -> Selecting Hero '{hero_name}' at {(cx, cy)}")
                self.tap_screen(cx, cy)
                time.sleep(0.06)
                pt = points[i % len(points)]
                print(f"     Deploying Hero '{hero_name}' at {pt} on {side}")
                self.tap_normalized(pt[0], pt[1])
                time.sleep(0.25)

    def monitor_battle(self, duration_sec: int) -> Dict[str, Any]:
        """
        Monitor live destruction percentage and stars during battle execution.
        """
        log = []
        for second in range(1, duration_sec + 1):
            time.sleep(1.0)
            frame = (
                self.controller.get_screenshot_fast()
                if self.use_fast_pipeline
                else self.controller.get_screenshot()
            )
            ui_state = self.ui_reader.parse_ui_fast(frame)
            log.append({
                "second": second,
                "destruction": ui_state["destruction_percentage"],
                "stars": ui_state["stars"],
            })
            print(
                f"  [T+{second:02d}s] Destruction: {ui_state['destruction_percentage']:.1f}% | "
                f"Stars: {ui_state['stars']}"
            )

        final_dest = log[-1]["destruction"] if log else 0.0
        final_stars = log[-1]["stars"] if log else 0
        return {
            "duration_sec": duration_sec,
            "final_destruction": final_dest,
            "final_stars": final_stars,
            "log": log,
        }


if __name__ == "__main__":
    attacker = PredeterminedAttacker(use_fast_pipeline=True)

    # Test 1: Electro Dragon Line Sweep on Bottom-Left side with all 4 Heroes alongside
    result_edrag = attacker.auto_attack(troop_type="EDRAGON", side="BOTTOM_LEFT", monitor_duration_sec=3)
    print("\nEDrag Line Sweep Test Result:", result_edrag)

    # Test 2: Sneaky Goblin Surround Attack on all 4 sides with 1 Hero per side
    result_gobs = attacker.auto_attack(troop_type="SNEAKY_GOBLIN", monitor_duration_sec=3)
    print("\nSneaky Goblin Surround Test Result:", result_gobs)
