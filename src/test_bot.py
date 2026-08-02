"""
Master Test Runner & Auto-Lobby System for BlueStacks 5 (BST 5)
---------------------------------------------------------------
Easy command-line tool to test ADB connection to BlueStacks 5 (default: 127.0.0.1:5555),
verify 1280x720 coordinates, check for errors, navigate the lobby using your UI buttons,
and launch predetermined attacks.
"""

import argparse
import sys
import time
from src.calibrate_coords import print_coordinate_tables, check_adb_connection, generate_calibration_overlay
from src.agent.predetermined_agent import PredeterminedAttacker
from src.vision.ui_matcher import UIMatcher


def run_auto_lobby(adb_serial: str, troop_type: str, side: str, duration: int, use_adb: bool = True):
    """
    Automates the full lobby-to-battle loop using attack.png, find.PNG, etc.
    """
    print("[AUTO-LOBBY] Initializing UIMatcher and Predetermined Attacker...")
    attacker = PredeterminedAttacker(use_fast_pipeline=not use_adb, device_serial=adb_serial)
    ui_matcher = UIMatcher()

    # Step 1: Look for Home Village Attack Button
    print("[AUTO-LOBBY] Searching for Home Village 'Attack' button (attack.png)...")
    for _ in range(5):
        frame = attacker.controller.get_screenshot() if use_adb else attacker.controller.get_screenshot_fast()
        pos = ui_matcher.find_button(frame, "attack")
        if pos:
            print(f"[AUTO-LOBBY] Found 'Attack' button at {pos}. Tapping...")
            attacker.tap_screen(pos[0], pos[1])
            time.sleep(1.5)
            break
        time.sleep(1.0)
    else:
        print("[AUTO-LOBBY] Notice: 'Attack' button not found on current screen. Continuing search for 'Find a Match'...")

    # Step 2: Look for 'Find a Match' or 'Attack Final' button
    print("[AUTO-LOBBY] Searching for 'Find a Match' (find.PNG) or 'Attack Final' (attack_final.PNG)...")
    for _ in range(5):
        frame = attacker.controller.get_screenshot() if use_adb else attacker.controller.get_screenshot_fast()
        pos_find = ui_matcher.find_button(frame, "find") or ui_matcher.find_button(frame, "attack_final")
        if pos_find:
            print(f"[AUTO-LOBBY] Found battle start button at {pos_find}. Tapping...")
            attacker.tap_screen(pos_find[0], pos_find[1])
            time.sleep(4.0)  # Wait for cloud search and enemy base to load
            break
        time.sleep(1.0)
    else:
        print("[AUTO-LOBBY] Notice: Could not find 'Find a Match' button. Assuming we are already in battle!")

    # Step 3: Execute the Predetermined Attack
    print(f"[AUTO-LOBBY] Starting battle deployment: {troop_type} (side: {side})...")
    result = attacker.auto_attack(troop_type=troop_type, side=side, monitor_duration_sec=duration)

    # Step 4: Check if 'Surrender / End Battle' button is visible at the end
    frame_end = attacker.controller.get_screenshot() if use_adb else attacker.controller.get_screenshot_fast()
    pos_surrender = ui_matcher.find_button(frame_end, "surrender")
    if pos_surrender:
        print(f"[AUTO-LOBBY] Battle ended or Surrender visible at {pos_surrender}. Tapping to return home...")
        attacker.tap_screen(pos_surrender[0], pos_surrender[1])

    return result


def run_tests(adb_serial: str, mode: str, troop_type: str, side: str, duration: int, use_adb: bool = False):
    print("=====================================================================")
    print("      CLASH-AI BLUESTACKS 5 (BST 5) TEST & LAUNCHER SYSTEM           ")
    print("=====================================================================")
    print(f"  Target Device/IP  : {adb_serial}")
    print(f"  Execution Mode    : {mode.upper()}")
    print(f"  Input Controller  : {'ADB Socket (127.0.0.1:5555)' if use_adb else 'Fast Window Capture / Click'}")
    print("=====================================================================\n")

    if mode == "calibrate":
        print_coordinate_tables(1280, 720)
        success, frame, w, h = check_adb_connection(adb_serial)
        generate_calibration_overlay(frame)
        print("\n[INFO] Calibration complete. Check 'debug_calibration_1280x720.png' to view tap overlays.")
        return

    if mode == "test-connection":
        print("[INFO] Running connection & screen capture test...")
        success, frame, w, h = check_adb_connection(adb_serial)
        if success:
            print("[SUCCESS] BlueStacks 5 is connected and responding to screen capture & input commands!")
        else:
            print("[ERROR] Could not capture live screen from BlueStacks 5.")
            print("  -> Try running: adb connect 127.0.0.1:5555")
        return

    if mode in ["attack", "auto-lobby"]:
        try:
            if mode == "auto-lobby":
                result = run_auto_lobby(
                    adb_serial=adb_serial,
                    troop_type=troop_type,
                    side=side,
                    duration=duration,
                    use_adb=use_adb,
                )
            else:
                attacker = PredeterminedAttacker(
                    use_fast_pipeline=not use_adb,
                    device_serial=adb_serial,
                )
                result = attacker.auto_attack(
                    troop_type=troop_type,
                    side=side,
                    monitor_duration_sec=duration,
                )

            print("\n=====================================================================")
            print("                       FINAL ATTACK RESULT                           ")
            print("=====================================================================")
            print(f"  Troop Type Deployed    : {troop_type}")
            print(f"  Final Destruction %    : {result['final_destruction']:.1f}%")
            print(f"  Final Stars Earned     : {result['final_stars']}")
            print(f"  Battle Duration        : {result['duration_sec']} seconds")
            print("=====================================================================\n")
        except Exception as e:
            print(f"\n[ERROR] Attack execution encountered an exception: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and Run Clash-AI Bot on BlueStacks 5")
    parser.add_argument(
        "--mode",
        type=str,
        default="test-connection",
        choices=["test-connection", "calibrate", "attack", "auto-lobby"],
        help="What to run: 'test-connection', 'calibrate', 'attack', or 'auto-lobby'",
    )
    parser.add_argument(
        "--adb-serial",
        type=str,
        default="127.0.0.1:5555",
        help="BlueStacks ADB serial number / localhost IP (default: 127.0.0.1:5555)",
    )
    parser.add_argument(
        "--troop",
        type=str,
        default="EDRAGON",
        choices=["EDRAGON", "DRAGON", "SNEAKY_GOBLIN", "VALKYRIE"],
        help="Troop type for attack mode (default: EDRAGON)",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="BOTTOM_LEFT",
        choices=["TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"],
        help="Side to attack from for line sweep attacks (default: BOTTOM_LEFT)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="How many seconds to monitor battle after deployment (default: 15)",
    )
    parser.add_argument(
        "--use-adb",
        action="store_true",
        help="Use direct ADB socket commands instead of window capture",
    )

    args = parser.parse_args()
    run_tests(
        adb_serial=args.adb_serial,
        mode=args.mode,
        troop_type=args.troop,
        side=args.side,
        duration=args.duration,
        use_adb=args.use_adb,
    )
