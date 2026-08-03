"""
1280x720 Coordinate Calibration & ADB Diagnostic Tool for BlueStacks 5
----------------------------------------------------------------------
Tests connection to BlueStacks 5 over ADB (default: 127.0.0.1:5555),
verifies 1280x720 resolution, prints an exact (X, Y) tap coordinate table,
and generates a visual overlay debug image ('debug_calibration_1280x720.png').
"""

import argparse
import os
import time
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from src.controller.adb_controller import ADBController
from src.vision.card_scanner import CardScanner


def print_coordinate_tables(width: int = 1280, height: int = 720) -> None:
    """Print exact (X, Y) pixel coordinates for 1280x720 resolution."""
    print("=====================================================================")
    print(f"      CLASH-AI EXACT 1280x720 TAP COORDINATE TABLE (BST 5)         ")
    print("=====================================================================\n")

    print("[1] DEPLOYMENT CARD BAR SLOTS (Bottom Bar Y = 655 px):")
    print("---------------------------------------------------------------------")
    print(f"  {'Slot':<10} | {'X Pixel':<12} | {'Y Pixel':<12} | {'Default Assignment':<20}")
    print("---------------------------------------------------------------------")

    scanner = CardScanner(screen_width=width, screen_height=height)
    default_assignments = [
        "SNEAKY_GOBLIN / VALKYRIE",
        "KING (Barbarian King)",
        "QUEEN (Archer Queen)",
        "WARDEN (Grand Warden)",
        "CHAMPION (Royal Champion)",
        "Spell / Troop Slot 6",
        "Spell / Troop Slot 7",
        "Spell / Troop Slot 8",
    ]

    for i in range(1, 9):
        x, y = scanner.get_slot_coordinate(i, width, height)
        assignment = default_assignments[i - 1] if i <= len(default_assignments) else f"Slot {i}"
        print(f"  Slot {i:<5} | X = {x:<8} | Y = {y:<8} | {assignment}")

    print("\n[2] BASE PERIMETER 4 SIDES (Normalized -> 1280x720 Pixel Coords):")
    print("---------------------------------------------------------------------")
    print(f"  {'Side Name':<15} | {'Start (X, Y)':<18} | {'Midpoint (Hero Drop)':<22} | {'End (X, Y)':<18}")
    print("---------------------------------------------------------------------")

    sides = {
        "TOP_LEFT": [(0.18, 0.45), (0.43, 0.20)],
        "TOP_RIGHT": [(0.57, 0.20), (0.82, 0.45)],
        "BOTTOM_RIGHT": [(0.82, 0.55), (0.57, 0.80)],
        "BOTTOM_LEFT": [(0.43, 0.80), (0.18, 0.55)],
    }

    for name, pts in sides.items():
        start_x, start_y = int(pts[0][0] * width), int(pts[0][1] * height)
        end_x, end_y = int(pts[1][0] * width), int(pts[1][1] * height)
        mid_x = (start_x + end_x) // 2
        mid_y = (start_y + end_y) // 2
        print(
            f"  {name:<15} | ({start_x:4d}, {start_y:4d})     | "
            f"({mid_x:4d}, {mid_y:4d})            | ({end_x:4d}, {end_y:4d})"
        )
    print("=====================================================================\n")


def check_adb_connection(adb_serial: str, adb_path: Optional[str] = None) -> Tuple[bool, np.ndarray, int, int]:
    """
    Test ADB connection to BlueStacks 5 and return status, screenshot frame, width, and height.
    """
    print(f"[INFO] Testing ADB connection to BlueStacks 5 at '{adb_serial}'...")
    adb = ADBController(device_serial=adb_serial, adb_path=adb_path)

    # Test resolution query
    w, h = adb.get_screen_resolution()
    print(f"[INFO] Detected screen resolution from device: {w}x{h}")
    if w != 1280 or h != 720:
        print(
            f"[WARN] Your device resolution is {w}x{h}, but 1280x720 is recommended for Clash of Clans! "
            f"Please check BlueStacks Settings -> Display -> 1280x720."
        )

    # Test screenshot capture
    try:
        frame = adb.get_screenshot()
        # Verify frame isn't empty black dummy frame
        if frame is not None and frame.shape[0] == h and frame.shape[1] == w and np.any(frame > 0):
            print(f"[SUCCESS] Successfully captured {w}x{h} live screenshot from BlueStacks 5!")
            return True, frame, w, h
        else:
            print("[WARN] Captured screenshot was blank/dummy. ADB screencap may have returned empty buffer.")
            return False, frame, 1280, 720
    except Exception as e:
        print(f"[ERROR] Failed to capture screenshot from BlueStacks 5: {e}")
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        return False, dummy, 1280, 720


def generate_calibration_overlay(frame: np.ndarray, save_path: str = "debug_calibration_1280x720.png") -> None:
    """
    Draw 1280x720 card slots and 4 sides on the screenshot frame and save it as an inspectable debug image.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Draw Card Bar Slots (1..8)
    scanner = CardScanner(screen_width=w, screen_height=h)
    for slot_idx in range(1, 9):
        x, y = scanner.get_slot_coordinate(slot_idx, w, h)
        cv2.circle(overlay, (x, y), 15, (0, 255, 0), -1)
        cv2.putText(
            overlay,
            f"S{slot_idx}",
            (x - 15, y - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Draw 4 Base Perimeter Sides
    side_colors = {
        "TOP_LEFT": (0, 255, 255),      # Yellow
        "TOP_RIGHT": (255, 0, 255),     # Magenta
        "BOTTOM_RIGHT": (0, 128, 255),  # Orange
        "BOTTOM_LEFT": (255, 255, 0),   # Cyan
    }
    sides = {
        "TOP_LEFT": [(0.18, 0.45), (0.23, 0.40), (0.28, 0.35), (0.33, 0.30), (0.38, 0.25), (0.43, 0.20)],
        "TOP_RIGHT": [(0.57, 0.20), (0.62, 0.25), (0.67, 0.30), (0.72, 0.35), (0.77, 0.40), (0.82, 0.45)],
        "BOTTOM_RIGHT": [(0.82, 0.55), (0.77, 0.60), (0.72, 0.65), (0.67, 0.70), (0.62, 0.75), (0.57, 0.80)],
        "BOTTOM_LEFT": [(0.43, 0.80), (0.38, 0.75), (0.33, 0.70), (0.28, 0.65), (0.23, 0.60), (0.18, 0.55)],
    }

    for name, pts in sides.items():
        color = side_colors[name]
        px_pts = [(int(p[0] * w), int(p[1] * h)) for p in pts]
        for i in range(len(px_pts) - 1):
            cv2.line(overlay, px_pts[i], px_pts[i + 1], color, 3)
        for pt in px_pts:
            cv2.circle(overlay, pt, 6, color, -1)
        # Label midpoint
        mid = px_pts[len(px_pts) // 2]
        cv2.putText(overlay, name, (mid[0] - 40, mid[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(save_path, overlay)
    print(f"[INFO] Calibration overlay image saved to '{save_path}'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate 1280x720 Coordinates & Test BlueStacks ADB")
    parser.add_argument("--adb-serial", type=str, default="127.0.0.1:5555", help="ADB Serial/IP for BlueStacks 5 (default: 127.0.0.1:5555)")
    parser.add_argument("--adb-path", type=str, default=None, help="Custom path to adb.exe or HD-Adb.exe")
    args = parser.parse_args()

    # 1. Print Coordinate Table
    print_coordinate_tables(1280, 720)

    # 2. Check BlueStacks ADB Connection
    success, frame, w, h = check_adb_connection(args.adb_serial, args.adb_path)

    # 3. Generate Debug Image
    generate_calibration_overlay(frame)

    if not success:
        print("\n[TIP] If BlueStacks ADB connection failed:")
        print("  1. Open BlueStacks 5 -> Settings (Gear Icon) -> Advanced -> Enable 'Android Debug Bridge (ADB)'.")
        print("  2. Verify the port (usually 5555 or 5554).")
        print("  3. Run: adb connect 127.0.0.1:5555")
