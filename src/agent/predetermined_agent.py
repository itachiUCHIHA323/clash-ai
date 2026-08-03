"""
Predetermined Attacking Engine (Precision Greyed-Out Deployment & Outermost Border Geometry)
-----------------------------------------------------------------------------------------
Automates specialized deployment strategies with strict 4-Step Tactical Execution:
1. STEP 1 (Troops First - ALL OF THEM): Selects troop cards and taps along the absolute
   outermost grass border (margin 10-12%) UNTIL THE CARD IS GREYED OUT (fully deployed).
2. STEP 2 (Then Heroes): Selects Hero cards after all troops are greyed out.
3. STEP 3 (Spells A BIT AHEAD - NOT AT BACK): Deploys Rage / Support spells slightly INWARD
   toward the center of the base (~18% ahead of the troop drop line), placing the spell
   circle right where troops walk into enemy defenses.
4. STEP 4 (Hero Ability Activation): Waits ~7 seconds after hero deployment and taps all
   Hero cards in the deployment bar again to activate Hero Abilities (Gauntlet/Tome/Arrow).
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from src.controller.fast_controller import FastController
from src.controller.adb_controller import ADBController
from src.vision.card_scanner import CardScanner
from src.vision.fast_ocr import FastUIReader


class PredeterminedAttacker:
    """
    Executes multi-wave tactical deployments: Troops -> Heroes -> Inward Rage -> Hero Abilities.
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

        # 4-Sided Base Perimeter Geometry (Absolute Outer Safe Grass Border 0.0 - 1.0, margin = 8%)
        # Positioned strictly on the outermost edge of the screen grass (8% margin) to NEVER touch the red restricted area
        self.sides_geometry = {
            "TOP_LEFT": [
                (0.08, 0.40), (0.14, 0.35), (0.20, 0.30), (0.26, 0.25),
                (0.32, 0.20), (0.38, 0.14)
            ],
            "TOP_RIGHT": [
                (0.62, 0.14), (0.68, 0.20), (0.74, 0.25), (0.80, 0.30),
                (0.86, 0.35), (0.92, 0.40)
            ],
            "BOTTOM_RIGHT": [
                (0.92, 0.60), (0.86, 0.65), (0.80, 0.70), (0.74, 0.75),
                (0.68, 0.80), (0.62, 0.86)
            ],
            "BOTTOM_LEFT": [
                (0.38, 0.86), (0.32, 0.80), (0.26, 0.75), (0.20, 0.70),
                (0.14, 0.65), (0.08, 0.60)
            ],
        }

        # Spell Geometry: Positioned ~18% inward toward center (0.50, 0.45) - ahead of troops, not at back
        self.spell_geometry = {
            "TOP_LEFT": [(0.25, 0.38), (0.32, 0.30), (0.38, 0.24)],
            "TOP_RIGHT": [(0.62, 0.24), (0.68, 0.30), (0.75, 0.38)],
            "BOTTOM_RIGHT": [(0.75, 0.52), (0.68, 0.60), (0.62, 0.68)],
            "BOTTOM_LEFT": [(0.38, 0.68), (0.32, 0.60), (0.25, 0.52)],
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

    def _get_card_info(self, map_data: Dict[str, Any], card_name: str, default_count: int = 28) -> Optional[Tuple[int, int, int]]:
        """Safely extract (x, y, count) for a card from army_map or card_map."""
        if card_name not in map_data:
            return None

        val = map_data[card_name]
        if isinstance(val, dict):
            pos = val.get("pos", (0, 0))
            count = val.get("count", default_count)
            return (int(pos[0]), int(pos[1]), int(count))
        elif isinstance(val, (tuple, list)) and len(val) >= 2:
            return (int(val[0]), int(val[1]), default_count)
        return None

    def zoom_out_base(self) -> None:
        """Perform pinch-out/zoom-out motion so the entire base is visible before deploying."""
        if hasattr(self.controller, "zoom_out"):
            self.controller.zoom_out()

    def _deploy_card_until_greyed_out(self, card_name: str, army_map: Dict[str, Any], target_sides: List[str]) -> bool:
        """
        Taps the card icon to select it, then repeatedly taps along the outermost grass border
        of target_sides UNTIL the card icon is greyed out (fully deployed / out of units).
        """
        info = self._get_card_info(army_map, card_name, default_count=24)
        if not info:
            return False

        cx, cy, initial_count = info
        print(f"  -> [DEPLOYING CARD] Selecting '{card_name}' at {(cx, cy)} (Initial count: {initial_count})")
        self.tap_screen(cx, cy)
        time.sleep(0.08)

        tapped_total = 0
        max_taps = max(40, initial_count + 15)  # Guard against infinite loop

        while tapped_total < max_taps:
            for side_name in target_sides:
                points = self.sides_geometry[side_name]
                pt = points[(tapped_total // max(1, len(target_sides))) % len(points)]
                self.tap_normalized(pt[0], pt[1])
                tapped_total += 1
                time.sleep(0.04)

            # Check if card has become empty / greyed out (all units deployed)
            if tapped_total % 4 == 0 or tapped_total >= initial_count:
                frame = (
                    self.controller.get_screenshot_fast()
                    if self.use_fast_pipeline
                    else self.controller.get_screenshot()
                )
                if self.card_scanner.is_card_empty(frame, cx, cy):
                    print(f"     [CARD EMPTY / GREYED OUT] '{card_name}' fully deployed after {tapped_total} taps!")
                    return True

        print(f"     [DEPLOY DONE] '{card_name}' finished after {tapped_total} taps.")
        return True

    def auto_attack(self, troop_type: str = "VALKYRIE", side: str = "BOTTOM_LEFT", monitor_duration_sec: int = 20) -> Dict[str, Any]:
        """
        Automatically execute 4-Step Tactical Attack:
        Troops first (until greyed out) -> Heroes -> Inward Rage ahead of troops -> Hero Abilities.
        """
        troop_type = troop_type.upper()
        print(f"\n=====================================================================")
        print(f"      4-STEP TACTICAL ATTACK ENGINE (Troop: {troop_type})           ")
        print("=====================================================================")

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
            print(f"[WARN] Unknown troop type '{troop_type}'. Using surround deployment.")
            self.execute_surround_attack(troop_type, army_map)

        print(f"[INFO] Deployment complete! Monitoring battle for {monitor_duration_sec}s...")
        return self.monitor_battle(monitor_duration_sec)

    def execute_surround_attack(self, troop_type: str, army_map: Dict[str, Any]) -> None:
        """
        4-Step Surround Attack (Valkyries / Sneaky Goblins):
        - Step 1: Deploy ALL troops across all 4 sides along outermost edge UNTIL GREYED OUT.
        - Step 2: Deploy Heroes (1 per side).
        - Step 3: Deploy Rage Spells slightly INWARD ahead of troops.
        - Step 4: Wait ~7s and tap Hero cards to activate Hero Abilities.
        """
        print(f"[SURROUND ATTACK] Zooming out base & preparing 4-Step Tactical Deployment...")
        self.zoom_out_base()
        time.sleep(0.3)

        # STEP 1: Deploy ALL troops first across all 4 sides UNTIL GREYED OUT
        print(f"[STEP 1: TROOPS FIRST] Deploying ALL '{troop_type}' across all 4 outermost edges until greyed out...")
        self._deploy_card_until_greyed_out(troop_type, army_map, list(self.sides_geometry.keys()))
        time.sleep(0.4)

        # STEP 2: Deploy Heroes AFTER troops (1 Hero on EACH side)
        print("[STEP 2: THEN HEROES] Deploying available Heroes (1 per side)...")
        hero_side_mapping = {
            "KING": "TOP_LEFT",
            "QUEEN": "TOP_RIGHT",
            "WARDEN": "BOTTOM_RIGHT",
            "CHAMPION": "BOTTOM_LEFT",
        }
        for hero_name, assigned_side in hero_side_mapping.items():
            h_info = self._get_card_info(army_map, hero_name, default_count=1)
            if h_info:
                cx, cy, _ = h_info
                self.tap_screen(cx, cy)
                time.sleep(0.08)
                points = self.sides_geometry[assigned_side]
                midpoint = points[len(points) // 2]
                print(f"  -> Dropping Hero '{hero_name}' at {assigned_side}")
                self.tap_normalized(midpoint[0], midpoint[1])
                time.sleep(0.15)

        time.sleep(0.4)

        # STEP 3: Deploy Spells A BIT AHEAD of troops toward center (not at back)
        self._deploy_spells_ahead(army_map, target_sides=list(self.spell_geometry.keys()))

        # STEP 4: Activate Hero Abilities after heroes enter combat
        self._schedule_hero_ability_activation(army_map, delay_sec=6.5)

    def execute_line_sweep_attack(self, troop_type: str, side: str, army_map: Dict[str, Any]) -> None:
        """
        4-Step Line Sweep Attack (Dragons / E-Dragons):
        - Step 1: Deploy ALL air troops along 1 outermost side UNTIL GREYED OUT.
        - Step 2: Deploy ALL Heroes alongside them on that exact same side.
        - Step 3: Deploy Rage Spells INWARD along the flight path ahead of dragons.
        - Step 4: Wait ~7s and tap Hero cards to activate Hero Abilities.
        """
        side = side.upper()
        if side not in self.sides_geometry:
            raise ValueError(f"Invalid side '{side}'. Must be one of {list(self.sides_geometry.keys())}")

        print(f"[LINE SWEEP ATTACK] Zooming out base & preparing 4-Step Tactical Deployment along {side}...")
        self.zoom_out_base()
        time.sleep(0.3)
        points = self.sides_geometry[side]

        # STEP 1: Deploy ALL troops first along selected side UNTIL GREYED OUT
        print(f"[STEP 1: TROOPS FIRST] Deploying ALL '{troop_type}' along outermost edge of side: {side} until greyed out...")
        self._deploy_card_until_greyed_out(troop_type, army_map, [side])
        time.sleep(0.4)

        # STEP 2: Deploy ALL Heroes alongside troops on the SAME side
        print(f"[STEP 2: THEN HEROES] Deploying ALL available HEROES alongside {troop_type} on side: {side}...")
        heroes = ["KING", "QUEEN", "WARDEN", "CHAMPION"]
        for i, hero_name in enumerate(heroes):
            h_info = self._get_card_info(army_map, hero_name, default_count=1)
            if h_info:
                cx, cy, _ = h_info
                self.tap_screen(cx, cy)
                time.sleep(0.08)
                pt = points[i % len(points)]
                print(f"  -> Dropping Hero '{hero_name}' at {pt}")
                self.tap_normalized(pt[0], pt[1])
                time.sleep(0.2)

        time.sleep(0.4)

        # STEP 3: Deploy Spells A BIT AHEAD of troops along the same side
        self._deploy_spells_ahead(army_map, target_sides=[side])

        # STEP 4: Activate Hero Abilities after combat begins
        self._schedule_hero_ability_activation(army_map, delay_sec=7.0)

    def _deploy_spells_ahead(self, army_map: Dict[str, Any], target_sides: List[str]) -> None:
        """
        Step 3: Checks for available Spells (RAGE, FREEZE, CLONE, etc.) in army_map
        and drops them 'a bit ahead of troops toward center - not at back'.
        """
        spell_names = ["RAGE", "FREEZE", "CLONE", "LIGHTNING", "JUMP", "EARTHQUAKE"]
        found_spells = [s for s in spell_names if s in army_map and self._get_card_info(army_map, s, 0)[2] > 0]
        if not found_spells:
            print("[STEP 3: SPELLS] No available spells found in deployment bar.")
            return

        print(f"[STEP 3: SPELLS AHEAD] Deploying spells ({found_spells}) A BIT AHEAD of troops toward core defenses...")
        for spell_name in found_spells:
            s_info = self._get_card_info(army_map, spell_name, default_count=2)
            if not s_info:
                continue
            cx, cy, count = s_info
            self.tap_screen(cx, cy)
            time.sleep(0.08)

            for i, side_name in enumerate(target_sides):
                if i >= count:
                    break
                spell_pts = self.spell_geometry[side_name]
                inward_pt = spell_pts[len(spell_pts) // 2]
                print(f"  -> Dropping Spell '{spell_name}' ahead on {side_name} at {inward_pt}")
                self.tap_normalized(inward_pt[0], inward_pt[1])
                time.sleep(0.15)

    def _schedule_hero_ability_activation(self, army_map: Dict[str, Any], delay_sec: float = 7.0) -> None:
        """
        Step 4: Waits until heroes engage enemy defenses, then taps all deployed Hero cards
        in the deployment bar to activate Hero Abilities (Iron Fist, Eternal Tome, etc.).
        """
        heroes = ["KING", "QUEEN", "WARDEN", "CHAMPION"]
        deployed_heroes = [h for h in heroes if h in army_map]
        if not deployed_heroes:
            print("[STEP 4: HERO ABILITY] No heroes were deployed to activate.")
            return

        print(f"[STEP 4: HERO ABILITY] Waiting {delay_sec:.1f}s for heroes to engage enemy defenses...")
        time.sleep(delay_sec)

        print(f"[STEP 4: HERO ABILITY] Tapping Hero cards ({deployed_heroes}) in deployment bar to trigger abilities!")
        for hero_name in deployed_heroes:
            h_info = self._get_card_info(army_map, hero_name, default_count=1)
            if h_info:
                cx, cy, _ = h_info
                print(f"  -> Triggering ability for '{hero_name}' at {(cx, cy)}")
                self.tap_screen(cx, cy)
                time.sleep(0.15)

    def monitor_battle(self, duration_sec: int) -> Dict[str, Any]:
        """Monitor live destruction percentage and stars during battle execution."""
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
    result = attacker.auto_attack(troop_type="VALKYRIE", monitor_duration_sec=3)
    print("\nSummary Result:", result)
