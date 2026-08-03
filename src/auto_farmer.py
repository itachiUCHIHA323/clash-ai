"""
Master Auto-Farming Loop (Continuous Lobby -> Loot OCR -> Edge Deploy -> Return Home)
-------------------------------------------------------------------------------------
Automates the full Clash of Clans farming loop:
1. Tap 'Attack' (attack.png) -> Tap 'Find a Match' (find.PNG).
2. Scan enemy base loot (Gold & Elixir) using OCR.
   - If Gold < 800,000 OR Elixir < 800,000 -> Tap 'Next' (next.PNG) and repeat check.
   - If Gold >= 800,000 AND Elixir >= 800,000 -> Proceed to attack!
3. Pinch-Out / Zoom Out so the whole base and outer corners are visible.
4. Scan troop cards (Valkyries, Dragons, E-Drags, Sneaky Goblins, Heroes).
5. Deploy along the OUTERMOST green edge to avoid the red 'cannot deploy here' error:
   - Valkyries / Sneaky Goblins -> All 4 sides + 1 Hero per side.
   - Dragons / E-Drags -> Single side + all 4 Heroes alongside on that same side.
6. Monitor battle until 'Return Home' (return.PNG / surrender.PNG) is visible -> Tap it -> Repeat!
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
        use_adb: bool = True,
    ):
        self.adb_serial = adb_serial
        self.adb_path = adb_path
        self.min_gold = min_gold
        self.min_elixir = min_elixir
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

    def search_and_attack_cycle(self, troop_type: str = "VALKYRIE", side: str = "BOTTOM_LEFT", battle_timeout_sec: int = 180) -> bool:
        """
        Execute one complete farming cycle: Lobby -> Loot Search -> Attack -> Return Home.
        """
        print("\n=====================================================================")
        print("                 STARTING NEW AUTO-FARMING CYCLE                     ")
        print("=====================================================================")

        # Step 1: Tap 'Attack' in Home Village (attack.png)
        print("[STEP 1] Searching for Home Village 'Attack' button (attack.png)...")
        for _ in range(6):
            frame = self.get_frame()
            pos = self.ui_matcher.find_button(frame, "attack")
            if pos:
                print(f"  -> Found 'Attack' button at {pos}. Tapping...")
                self.tap(pos[0], pos[1])
                time.sleep(1.2)
                break
            time.sleep(1.0)
        else:
            print("[WARN] 'Attack' button not found. Checking if we are already in the match finder...")

        # Step 2: Tap 'Find a Match' (find.PNG)
        print("[STEP 2] Searching for 'Find a Match' (find.PNG)...")
        for _ in range(6):
            frame = self.get_frame()
            pos = self.ui_matcher.find_button(frame, "find") or self.ui_matcher.find_button(frame, "attack_final")
            if pos:
                print(f"  -> Found 'Find a Match' button at {pos}. Tapping...")
                self.tap(pos[0], pos[1])
                time.sleep(4.5)  # Wait for cloud search and enemy base to load
                break
            time.sleep(1.0)

        # Step 3: Loot Search Loop (OCR Gold & Elixir vs 800,000 threshold)
        print(f"[STEP 3] Entering Loot Search Loop (Min Gold: {self.min_gold:,} | Min Elixir: {self.min_elixir:,})...")
        max_skips = 20
        for skip_count in range(max_skips):
            frame = self.get_frame()
            loot = self.loot_reader.read_loot(frame)
            if self.loot_reader.is_loot_sufficient(loot):
                print(f"[SUCCESS] Base meets loot threshold on attempt #{skip_count + 1}! Preparing attack...")
                break
            else:
                pos_next = self.ui_matcher.find_button(frame, "next")
                if pos_next:
                    print(f"  -> Loot below 800k threshold. Tapping 'Next' (next.PNG) at {pos_next}...")
                    self.tap(pos_next[0], pos_next[1])
                    time.sleep(4.5)  # Wait for next base to load
                else:
                    print("[WARN] 'Next' button not found. Assuming ready to attack.")
                    break

        # Step 4: Pinch-Out / Zoom Out so all 4 edges of base are visible
        print("[STEP 4] Pinch-Out / Zoom Out motion so outermost base border is visible...")
        self.attacker.zoom_out_base()
        time.sleep(0.5)

        # Step 5: Execute Edge Deployment based on troop_type
        print(f"[STEP 5] Executing Outermost Edge Deployment for '{troop_type}'...")
        card_map = self.attacker.scan_battle_cards()
        print(f"  -> Scanned Card Map from deployment bar: {list(card_map.keys())}")

        if troop_type in ["VALKYRIE", "SNEAKY_GOBLIN"]:
            self.attacker.execute_surround_attack(troop_type, card_map)
        elif troop_type in ["DRAGON", "EDRAGON"]:
            self.attacker.execute_line_sweep_attack(troop_type, side, card_map)
        else:
            print(f"[WARN] Unknown troop type '{troop_type}'. Using surround deployment.")
            self.attacker.execute_surround_attack(troop_type, card_map)

        # Step 6: Battle Monitoring & Return Home Loop
        print("[STEP 6] Attack deployed! Monitoring battle and scanning for 'Return Home' (return.PNG)...")
        start_t = time.time()
        while time.time() - start_t < battle_timeout_sec:
            time.sleep(2.5)
            frame = self.get_frame()
            # Check for Return Home (return.PNG) or Surrender (surrender.PNG)
            pos_ret = self.ui_matcher.find_button(frame, "return") or self.ui_matcher.find_button(frame, "surrender")
            if pos_ret:
                print(f"[STEP 6] Battle finished or 'Return Home' visible at {pos_ret}. Tapping to return to village...")
                self.tap(pos_ret[0], pos_ret[1])
                time.sleep(5.0)  # Wait for home village to load
                return True

            ui_state = self.attacker.ui_reader.parse_ui_fast(frame)
            elapsed = int(time.time() - start_t)
            print(f"  [T+{elapsed:03d}s] Destruction: {ui_state['destruction_percentage']:.1f}% | Stars: {ui_state['stars']}")

        print("[STEP 6] Battle timeout reached. Checking once more for return button...")
        frame = self.get_frame()
        pos_ret = self.ui_matcher.find_button(frame, "return") or self.ui_matcher.find_button(frame, "surrender")
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
                self.search_and_attack_cycle(troop_type=troop_type, side=side)
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
    parser.add_argument("--cycles", type=int, default=5, help="Number of continuous farming attacks to run (default: 5)")
    parser.add_argument("--use-window", action="store_true", help="Use fast window capture instead of ADB")

    args = parser.parse_args()
    farmer = AutoFarmer(
        adb_serial=args.adb_serial,
        adb_path=args.adb_path,
        min_gold=args.min_gold,
        min_elixir=args.min_elixir,
        use_adb=not args.use_window,
    )
    farmer.run_continuous(troop_type=args.troop, side=args.side, max_cycles=args.cycles)
