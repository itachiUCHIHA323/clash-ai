"""
Interactive Loot OCR Diagnostic & Calibration Script
----------------------------------------------------
Captures a screenshot from BlueStacks 5 / MuMu Player, runs LootReader,
prints a detailed diagnostic table of extracted Gold, Elixir, and Dark Elixir,
and saves annotated debug images ('debug_loot_screen_boxes.png', 'debug_loot_gold_roi.png').
"""

import argparse
import cv2
import numpy as np
from src.controller.adb_controller import ADBController
from src.controller.fast_controller import FastController
from src.vision.loot_reader import LootReader


def run_loot_diagnostic(adb_serial: str = "127.0.0.1:5555", adb_path: str = None, use_adb: bool = True):
    print("=====================================================================")
    print("           CLASH-AI LOOT OCR DIAGNOSTIC & CALIBRATION               ")
    print("=====================================================================")

    controller = (
        ADBController(device_serial=adb_serial, adb_path=adb_path)
        if use_adb
        else FastController()
    )

    print("[INFO] Capturing screenshot from emulator...")
    frame = controller.get_screenshot() if use_adb else controller.get_screenshot_fast()
    h, w, _ = frame.shape
    print(f"[INFO] Frame captured ({w}x{h}). Initializing LootReader...")

    reader = LootReader(min_gold=800000, min_elixir=800000)

    print("\n[STEP 1] Running read_loot(frame) with debug ROI saving enabled...")
    loot_result = reader.read_loot(frame, save_debug_roi=True)

    print("\n=====================================================================")
    print("                     LOOT SCANNER RESULTS                            ")
    print("=====================================================================")
    print(f"  Available GOLD        : {loot_result['gold']:,}")
    print(f"  Available ELIXIR      : {loot_result['elixir']:,}")
    print(f"  Available DARK ELIXIR : {loot_result['dark_elixir']:,}")
    print("---------------------------------------------------------------------")
    print("  Debug Slice Images Saved to Repository Root:")
    print("    - debug_loot_gold_roi.png")
    print("    - debug_loot_elixir_roi.png")
    print("    - debug_loot_dark_elixir_roi.png")
    print("=====================================================================\n")

    # Draw visual boxes on copy of frame
    overlay = frame.copy()
    y1, y2 = int(h * 0.01), int(h * 0.36)
    x1, x2 = int(w * 0.01), int(w * 0.28)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(overlay, "Upper-Left Loot Search Area", (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imwrite("debug_loot_screen_boxes.png", overlay)
    print("[INFO] Saved screen search area overview to 'debug_loot_screen_boxes.png'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and inspect Upper-Left Loot OCR scanner")
    parser.add_argument("--adb-serial", type=str, default="127.0.0.1:5555", help="ADB Serial/IP")
    parser.add_argument("--adb-path", type=str, default=None, help="Custom path to adb.exe")
    parser.add_argument("--use-window", action="store_true", help="Use window capture instead of ADB")
    args = parser.parse_args()

    run_loot_diagnostic(adb_serial=args.adb_serial, adb_path=args.adb_path, use_adb=not args.use_window)
