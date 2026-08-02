"""
ADB Controller Module
---------------------
Provides an interface to connect to an Android device/emulator via ADB,
capture high-speed screen frames, and send simulated touchscreen taps and drags.
"""

import subprocess
import time
import cv2
import numpy as np
from typing import Tuple, Optional


class ADBController:
    def __init__(self, device_serial: Optional[str] = None):
        """
        Initialize the ADB controller.
        
        :param device_serial: Optional serial number of the target device/emulator
                              (e.g., 'emulator-5554' or '127.0.0.1:5555').
        """
        self.device_serial = device_serial
        self._adb_base = ["adb"]
        if self.device_serial:
            self._adb_base.extend(["-s", self.device_serial])

    def _run_adb(self, args: list, raw: bool = False):
        """Execute an ADB command."""
        cmd = self._adb_base + args
        if raw:
            return subprocess.check_output(cmd)
        else:
            return subprocess.run(cmd, capture_output=True, text=True, check=True)

    def get_screenshot(self) -> np.ndarray:
        """
        Capture a screenshot from the emulator/device and return it as a BGR OpenCV image.
        
        :return: np.ndarray of shape (H, W, 3) in BGR format.
        """
        try:
            raw_png = self._run_adb(["exec-out", "screencap", "-p"], raw=True)
            image_array = np.frombuffer(raw_png, np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("Failed to decode screenshot from ADB buffer.")
            return frame
        except Exception as e:
            # Fallback black image if ADB fails during offline/test mode
            print(f"[WARN] ADB screenshot failed ({e}). Returning blank 720x1280 test frame.")
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    def tap(self, x: int, y: int) -> None:
        """
        Simulate a touchscreen tap at coordinates (x, y).
        
        :param x: X coordinate in screen pixels.
        :param y: Y coordinate in screen pixels.
        """
        try:
            self._run_adb(["shell", "input", "tap", str(int(x)), str(int(y))])
        except Exception as e:
            print(f"[WARN] ADB tap({x}, {y}) failed: {e}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        """
        Simulate a swipe/drag gesture from (x1, y1) to (x2, y2).
        
        :param duration_ms: Duration of the swipe in milliseconds.
        """
        try:
            self._run_adb([
                "shell", "input", "swipe",
                str(int(x1)), str(int(y1)),
                str(int(x2)), str(int(y2)),
                str(int(duration_ms))
            ])
        except Exception as e:
            print(f"[WARN] ADB swipe failed: {e}")

    def get_screen_resolution(self) -> Tuple[int, int]:
        """
        Query the screen resolution (width, height) of the connected device.
        """
        try:
            output = self._run_adb(["shell", "wm", "size"]).stdout
            # Example output: "Physical size: 1280x720"
            parts = output.strip().split(":")[-1].strip().split("x")
            return int(parts[0]), int(parts[1])
        except Exception:
            return 1280, 720
