"""
Predetermined Scripted Attacking Agent
--------------------------------------
Executes structured, rule-based deployment strategies (e.g., BARCH Wave,
Giants + Wizards Funnel, Surround Spam) to test input reliability and establish
a benchmark before training Reinforcement Learning models.
"""

import time
from typing import List, Tuple, Dict, Any, Optional
from src.controller.fast_controller import FastController
from src.controller.adb_controller import ADBController
from src.vision.fast_ocr import FastUIReader


class ScriptedAttacker:
    """
    A rule-based scripted attacker that deploys troops in predetermined sequences
    and monitors real-time destruction percentage and stars.
    """

    def __init__(self, use_fast_pipeline: bool = True, device_serial: Optional[str] = None):
        self.use_fast_pipeline = use_fast_pipeline
        if self.use_fast_pipeline:
            self.controller = FastController()
            self.ui_reader = FastUIReader()
        else:
            self.controller = ADBController(device_serial=device_serial)
            self.ui_reader = FastUIReader()

    def get_screen_dimensions(self) -> Tuple[int, int]:
        """Return width and height of the active target screen/window."""
        if self.use_fast_pipeline:
            return self.controller.region["width"], self.controller.region["height"]
        else:
            return self.controller.get_screen_resolution()

    def tap(self, x: int, y: int) -> None:
        """Send a tap command via the active controller."""
        if self.use_fast_pipeline:
            self.controller.tap_fast(x, y)
        else:
            self.controller.tap(x, y)

    def select_card(self, card_idx: int) -> None:
        """
        Tap a deployment card in the bottom troop bar (1-indexed: 1..8).
        """
        w, h = self.get_screen_dimensions()
        # Cards are spaced evenly along the bottom bar (~15% to 85% width at 90% height)
        card_x = int(w * (0.15 + (card_idx - 1) * 0.10))
        card_y = int(h * 0.90)
        self.tap(card_x, card_y)
        time.sleep(0.05)

    def deploy_at_normalized(self, norm_x: float, norm_y: float) -> None:
        """
        Deploy selected unit at normalized (0.0 - 1.0) battle screen coordinates.
        """
        w, h = self.get_screen_dimensions()
        x = int(norm_x * w)
        y = int(norm_y * h)
        self.tap(x, y)

    def run_strategy(self, strategy_name: str = "BARCH_WAVE", monitor_duration_sec: int = 15) -> Dict[str, Any]:
        """
        Execute a predetermined attack deployment sequence and monitor results.

        :param strategy_name: Strategy selection ('BARCH_WAVE', 'GIANT_WIZARD_FUNNEL', 'SURROUND_SPAM').
        :param monitor_duration_sec: How long to monitor battle progress after deployment.
        """
        print(f"[INFO] Starting Predetermined Strategy: {strategy_name}")
        start_time = time.time()

        if strategy_name == "BARCH_WAVE":
            self._execute_barch_wave()
        elif strategy_name == "GIANT_WIZARD_FUNNEL":
            self._execute_giant_wizard_funnel()
        elif strategy_name == "SURROUND_SPAM":
            self._execute_surround_spam()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        print(f"[INFO] Deployment complete. Monitoring battle state for {monitor_duration_sec}s...")
        progress_log = []

        for i in range(monitor_duration_sec):
            time.sleep(1.0)
            frame = (
                self.controller.get_screenshot_fast()
                if self.use_fast_pipeline
                else self.controller.get_screenshot()
            )
            ui_state = self.ui_reader.parse_ui_fast(frame)
            progress_log.append({
                "second": i + 1,
                "destruction": ui_state["destruction_percentage"],
                "stars": ui_state["stars"],
            })
            print(
                f"  -> [T+{i+1:02d}s] Destruction: {ui_state['destruction_percentage']:.1f}% | "
                f"Stars: {ui_state['stars']}"
            )

        print("[INFO] Attack monitoring finished.")
        return {
            "strategy": strategy_name,
            "duration_sec": monitor_duration_sec,
            "final_destruction": progress_log[-1]["destruction"] if progress_log else 0.0,
            "final_stars": progress_log[-1]["stars"] if progress_log else 0,
            "log": progress_log,
        }

    def _execute_barch_wave(self) -> None:
        """
        BARCH Wave Strategy (Barbarians + Archers):
        1. Select Barbarians (Card 1) and deploy a line along the bottom-left edge to tank.
        2. Wait 1 second for defenses to lock onto Barbarians.
        3. Select Archers (Card 2) and deploy a line behind the Barbarians for ranged DPS.
        """
        print("  [STEP 1] Deploying tanking Barbarian line (Card 1) on bottom-left border...")
        self.select_card(1)
        # Drop 8 barbarians along bottom-left border
        for step in range(8):
            t = step / 7.0
            norm_x = 0.15 + t * 0.35  # from x=0.15 to x=0.50
            norm_y = 0.75 - t * 0.35  # from y=0.75 to y=0.40
            self.deploy_at_normalized(norm_x, norm_y)
            time.sleep(0.08)

        print("  [STEP 2] Waiting 1.0s for defenses to aggro Barbarians...")
        time.sleep(1.0)

        print("  [STEP 3] Deploying ranged Archer wave (Card 2) behind Barbarians...")
        self.select_card(2)
        for step in range(8):
            t = step / 7.0
            norm_x = 0.12 + t * 0.35
            norm_y = 0.78 - t * 0.35
            self.deploy_at_normalized(norm_x, norm_y)
            time.sleep(0.08)

    def _execute_giant_wizard_funnel(self) -> None:
        """
        Giant + Wizard Funnel Strategy:
        1. Deploy 3 Giants (Card 1) at a single focal point to absorb defense fire.
        2. Deploy Wizards (Card 2) on the left and right flanks to clear outside buildings (funneling).
        3. Deploy Wall Breakers / Heroes (Card 3) in the center.
        """
        print("  [STEP 1] Deploying 3 Giants (Card 1) at focal tanking point...")
        self.select_card(1)
        for _ in range(3):
            self.deploy_at_normalized(0.30, 0.60)
            time.sleep(0.15)

        print("  [STEP 2] Deploying flanking Wizards (Card 2) to funnel outside buildings...")
        self.select_card(2)
        # Left flank
        self.deploy_at_normalized(0.20, 0.68)
        time.sleep(0.1)
        self.deploy_at_normalized(0.22, 0.66)
        time.sleep(0.1)
        # Right flank
        self.deploy_at_normalized(0.40, 0.52)
        time.sleep(0.1)
        self.deploy_at_normalized(0.42, 0.50)

    def _execute_surround_spam(self) -> None:
        """
        Surround Spam Strategy:
        Quickly deploys units in a 360-degree circle around the perimeter of the base.
        """
        print("  [STEP 1] Deploying 360-degree surround perimeter attack (Card 1)...")
        self.select_card(1)
        perimeter_coords = [
            (0.20, 0.40), (0.30, 0.30), (0.45, 0.20), (0.60, 0.25),
            (0.75, 0.40), (0.75, 0.55), (0.65, 0.70), (0.50, 0.75),
            (0.35, 0.75), (0.22, 0.60),
        ]
        for nx, ny in perimeter_coords:
            self.deploy_at_normalized(nx, ny)
            time.sleep(0.08)


if __name__ == "__main__":
    attacker = ScriptedAttacker(use_fast_pipeline=True)
    result = attacker.run_strategy(strategy_name="BARCH_WAVE", monitor_duration_sec=5)
    print("\nSummary Result:", result)
