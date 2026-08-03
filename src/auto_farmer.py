"""
Master Auto-Farming Loop (State-Verified Lobby -> Loot OCR -> Edge Deploy -> Return Home)
---------------------------------------------------------------------------------------
Automates the full Clash of Clans farming loop with strict screen-state verification:
1. Tap 'Attack' (attack.png)
2. Tap 'Find a Match' (find.PNG)
3. Tap 'Attack Final' confirmation button (attack_final.PNG)
4. Verify we are on the Enemy Scout Screen (next.PNG MUST be visible!)
5. Scan enemy base loot (Gold & Elixir) using real OCR.
   - If Gold < 800,000 OR Elixir < 800,000 -> Tap 'Next' (next.PNG) and check next base.
   - If Gold >= 800,000 AND Elixir >= 800,000 -> Proceed to attack!
6. Pinch-Out / Zoom Out so the whole base and outer corners are visible.
7. Scan troop cards (Valkyries, Dragons, E-Drags, Sneaky Goblins, Heroes).
8. Deploy along the OUTERMOST green edge to avoid the red 'cannot deploy here' error.
9. Monitor battle until 'Return Home' (return.PNG / surrender.PNG) is visible -> Tap it -> Repeat!
"""

import argparse
import sys
import time
from typing import Dict, Any, Optional
from src.agent.predetermined_agent import PredeterminedAttacker
from src.vision.ui_matcher import UIMatcher
from src.vision.loot_reader import LootReader


class AutoFarmer:
    def __init__(
        self,
        adb_serial: str = "127.0.0.1:5555",
        adb_path: Optional[str] = None,
        min_gold: int = 800000,
        min_elixir: int = 800000,
        force_attack: bool = False,
        use_adb: bool = True,
    ):
        self.adb_serial = adb_serial
        self.adb_path = adb_path
        self.min_gold = min_gold
        self.min_elixir = min_elixir
        self.force_attack = force_attack
        self.use_adb = use_adb

        print("[AUTO-FARMER] Initializing Attacker, UI Matcher, and Loot OCR Scanner...")
        self.attacker = PredeterminedAttacker(
            use_fast_pipeline=not self.use_adb,
            device_serial=self.adb_serial,
            adb_path=self.adb_path,
        )
        self.ui_matcher = UIMatcher(template_paths=[".", "templates/ui"])
        self.loot_reader = LootReader(min_gold=self.min_gold, min_elixir=self.min_elixir)

    def get_frame(self):
        """Capture screen from ADB or Fast window."""
        return self.attacker.controller.get_screenshot() if self.use_adb else self.attacker.controller.get_screenshot_fast()

    def tap(self, x: int, y: int):
        """Send screen tap."""
        self.attacker.tap_screen(x, y)

    def _wait_and_tap(self, button_name: str, timeout_sec: int = 12, threshold: float = 0.68) -> bool:
        """
        Poll the screen until the requested button is visible, then tap it.
        Returns True if tapped, False if timed out.
        """
        start_t = time.time()
        while time.time() - start_t < timeout_sec:
            frame = self.get_frame()
            pos = self.ui_matcher.find_button(frame, button_name, threshold=threshold)
            if pos:
                print(f"  -> Found UI button '{button_name}' at {pos}. Tapping...")
                self.tap(pos[0], pos[1])
                time.sleep(1.2)
                return True
            time.sleep(0.5)
        print(f"[WARN] Timed out waiting for UI button '{button_name}'.")
        return False

    def _wait_for_button_visible(self, button_name: str, timeout_sec: int = 20, threshold: float = 0.68) -> bool:
        """
        Poll the screen until the requested button appears (without tapping it).
        Used to guarantee we are actually on the Enemy Scout screen (next.PNG).
        """
        start_t = time.time()
        while time.time() - start_t < timeout_sec:
            frame = self.get_frame()
            if self.ui_matcher.is_visible(frame, button_name, threshold=threshold):
                return True
            time.sleep(0.5)
        return False

    def search_and_attack_cycle(self, troop_type: str = "VALKYRIE", side: str = "BOTTOM_LEFT", battle_timeout_sec: int = 180) -> bool:
        """
        Execute one complete farming cycle with strict screen verification.
        """
        print("\n=====================================================================")
        print("                 STARTING NEW AUTO-FARMING CYCLE                     ")
        print("=====================================================================")

        # Step 1: Tap 'Attack' in Home Village (attack.png)
        print("[STEP 1] Searching for Home Village 'Attack' button (attack.png)...")
        if not self._wait_and_tap("attack", timeout_sec=10):
            print("[INFO] 'attack.png' not found. Checking if already inside matchmaking lobby...")

        # Step 2: Tap 'Find a Match' (find.PNG)
        print("[STEP 2] Searching for 'Find a Match' button (find.PNG)...")
        if not self._wait_and_tap("find", timeout_sec=10):
            print("[INFO] 'find.PNG' not found. Checking if already on confirmation screen...")

        # Step 3: Tap 'Attack Final' confirmation button (attack_final.PNG)
        print("[STEP 3] Searching for confirm 'Attack!' button (attack_final.PNG)...")
        if not self._wait_and_tap("attack_final", timeout_sec=10):
            print("[INFO] 'attack_final.PNG' not found. Verifying if enemy scout screen is active...")

        # Step 4: STRICT SAFETY GUARD — verify 'Next' button (next.PNG) is visible before doing ANYTHING!
        print("[STEP 4] Verifying Enemy Scout screen is loaded (waiting for next.PNG to appear)...")
        if not self._wait_for_button_visible("next", timeout_sec=20):
            print("[ERROR] Could not confirm we are on an enemy scout screen ('next.PNG' never appeared). Aborting cycle to prevent accidental lobby clicks!")
            return False
        print("[SUCCESS] Enemy Scout screen verified! We are looking at a real opponent base.")

        # Step 5: Loot Search Loop (OCR Gold & Elixir vs threshold)
        print(f"[STEP 5] Entering Loot Search Loop (Min Gold: {self.min_gold:,} | Min Elixir: {self.min_elixir:,})...")
        max_skips = 25
        for skip_count in range(max_skips):
            frame = self.get_frame()
            loot = self.loot_reader.read_loot(frame)
            if self.loot_reader.is_loot_sufficient(loot, force_attack=self.force_attack):
                print(f"[SUCCESS] Base #{skip_count + 1} meets loot threshold! Preparing attack...")
                break
            else:
                pos_next = self.ui_matcher.find_button(frame, "next", threshold=0.68)
                if pos_next:
                    print(f"  -> Loot below threshold. Tapping 'Next' (next.PNG) at {pos_next}...")
                    self.tap(pos_next[0], pos_next[1])
                    time.sleep(3.5)  # Wait for next scout base to settle
                    # Ensure next base has finished loading
                    self._wait_for_button_visible("next", timeout_sec=15)
                else:
                    print("[WARN] 'Next' button not found. Proceeding to attack.")
                    break

        # Step 6: Pinch-Out / Zoom Out so all 4 edges of base are visible
        print("[STEP 6] Pinch-Out / Zoom Out motion so outermost base border is visible...")
        self.attacker.zoom_out_base()
        time.sleep(0.5)

        # Step 7: Execute Edge Deployment based on troop_type
        print(f"[STEP 7] Executing Outermost Edge Deployment for '{troop_type}'...")
        card_map = self.attacker.scan_battle_cards()
        print(f"  -> Scanned Card Map from deployment bar: {list(card_map.keys())}")

        if troop_type in ["VALKYRIE", "SNEAKY_GOBLIN"]:
            self.attacker.execute_surround_attack(troop_type, card_map)
        elif troop_type in ["DRAGON", "EDRAGON"]:
            self.attacker.execute_line_sweep_attack(troop_type, side, card_map)
        else:
            print(f"[WARN] Unknown troop type '{troop_type}'. Using surround deployment.")
            self.attacker.execute_surround_attack(troop_type, card_map)

        # Step 8: Battle Monitoring & Return Home Loop
        print("[STEP 8] Attack deployed! Monitoring battle and scanning for 'Return Home' (return.PNG)...")
        start_t = time.time()
        while time.time() - start_t < battle_timeout_sec:
            time.sleep(2.0)
            frame = self.get_frame()
            pos_ret = (
                self.ui_matcher.find_button(frame, "return", threshold=0.68)
                or self.ui_matcher.find_button(frame, "surrender", threshold=0.68)
            )
            if pos_ret:
                print(f"[STEP 8] Battle finished or 'Return Home' visible at {pos_ret}. Tapping to return to village...")
                self.tap(pos_ret[0], pos_ret[1])
                time.sleep(4.5)  # Wait for home village to load
                return True

            ui_state = self.attacker.ui_reader.parse_ui_fast(frame)
            elapsed = int(time.time() - start_t)
            print(f"  [T+{elapsed:03d}s] Destruction: {ui_state['destruction_percentage']:.1f}% | Stars: {ui_state['stars']}")

        print("[STEP 8] Battle timeout reached. Checking once more for return button...")
        frame = self.get_frame()
        pos_ret = (
            self.ui_matcher.find_button(frame, "return", threshold=0.68)
            or self.ui_matcher.find_button(frame, "surrender", threshold=0.68)
        )
        if pos_ret:
            self.tap(pos_ret[0], pos_ret[1])
            time.sleep(4.0)

        return True

    def run_continuous(self, troop_type: str = "VALKYRIE", side: str = "BOTTOM_LEFT", max_cycles: int = 5):
        """
        Run continuous farming loop for max_cycles.
        """
        for cycle in range(1, max_cycles + 1):
            print(f"\n=====================================================================")
            print(f"                   AUTO-FARMING CYCLE {cycle} OF {max_cycles}        ")
            print("=====================================================================")
            try:
                success = self.search_and_attack_cycle(troop_type=troop_type, side=side)
                if not success:
                    print(f"[WARN] Cycle #{cycle} aborted due to state check failure. Waiting 5s...")
                    time.sleep(5.0)
                    continue
                print(f"[SUCCESS] Completed farming cycle #{cycle}. Waiting 3s before next cycle...")
                time.sleep(3.0)
            except Exception as e:
                print(f"[ERROR] Exception during farming cycle #{cycle}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Continuous Auto-Farming Loop for Clash of Clans")
    parser.add_argument("--adb-serial", type=str, default="127.0.0.1:5555", help="ADB Serial/IP (default: 127.0.0.1:5555)")
    parser.add_argument("--adb-path", type=str, default=None, help="Custom path to HD-Adb.exe or adb.exe")
    parser.add_argument("--troop", type=str, default="VALKYRIE", choices=["VALKYRIE", "SNEAKY_GOBLIN", "DRAGON", "EDRAGON"], help="Troop type to deploy")
    parser.add_argument("--side", type=str, default="BOTTOM_LEFT", choices=["TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"], help="Side for line sweep")
    parser.add_argument("--min-gold", type=int, default=800000, help="Minimum Gold threshold (default: 800,000)")
    parser.add_argument("--min-elixir", type=int, default=800000, help="Minimum Elixir threshold (default: 800,000)")
    parser.add_argument("--force-attack", action="store_true", help="Bypass loot OCR threshold check for manual attack testing")
    parser.add_argument("--cycles", type=int, default=5, help="Number of continuous farming attacks to run (default: 5)")
    parser.add_argument("--use-window", action="store_true", help="Use fast window capture instead of ADB")

    args = parser.parse_args()
    farmer = AutoFarmer(
        adb_serial=args.adb_serial,
        adb_path=args.adb_path,
        min_gold=args.min_gold,
        min_elixir=args.min_elixir,
        force_attack=args.force_attack,
        use_adb=not args.use_window,
    )
    farmer.run_continuous(troop_type=args.troop, side=args.side, max_cycles=args.cycles)
