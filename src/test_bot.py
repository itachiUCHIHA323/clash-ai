"""
Master Test Runner for BlueStacks 5 (BST 5)
-------------------------------------------
Easy command-line tool to test ADB connection to BlueStacks 5 (default: 127.0.0.1:5555),
verify 1280x720 coordinates, check for errors, and launch predetermined attacks.
"""

import argparse
import sys
import time
from src.calibrate_coords import print_coordinate_tables, check_adb_connection, generate_calibration_overlay
from src.agent.predetermined_agent import PredeterminedAttacker


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

    if mode == "attack":
        print(f"[INFO] Initializing Predetermined Attacker for troop: '{troop_type}' on side: '{side}'...")
        # If use_adb is True, pass use_fast_pipeline=False to use ADB over socket
        attacker = PredeterminedAttacker(
            use_fast_pipeline=not use_adb,
            device_serial=adb_serial,
        )

        try:
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
        choices=["test-connection", "calibrate", "attack"],
        help="What to run: 'test-connection', 'calibrate', or 'attack'",
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
