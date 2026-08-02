"""
ADB Controller Module (Windows & BlueStacks 5 Auto-Discovery)
-------------------------------------------------------------
Provides an interface to connect to an Android device/emulator via ADB,
capture high-speed screen frames, and send simulated touchscreen taps and drags.

Automatically discovers BlueStacks 5 HD-Adb.exe on Windows if 'adb' is not in PATH.
"""

import os
import shutil
import subprocess
import time
import cv2
import numpy as np
from typing import Tuple, Optional, List


class ADBController:
    def __init__(self, device_serial: Optional[str] = None, adb_path: Optional[str] = None):
        """
        Initialize the ADB controller.

        :param device_serial: Optional serial number of the target device/emulator
                              (e.g., '127.0.0.1:5555' for BlueStacks 5).
        :param adb_path: Custom path to adb.exe or HD-Adb.exe.
        """
        self.device_serial = device_serial
        self.adb_exe = self._find_adb_executable(adb_path)
        self._adb_base = [self.adb_exe]
        if self.device_serial:
            self._adb_base.extend(["-s", self.device_serial])

    def _find_adb_executable(self, custom_path: Optional[str] = None) -> str:
        """
        Locate a valid ADB executable on Windows, Linux, or macOS.
        Checks custom path, system PATH, and BlueStacks 5 default installation folders.
        """
        if custom_path and os.path.exists(custom_path):
            return custom_path

        # Check environment variable
        env_path = os.environ.get("ADB_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # Check system PATH
        which_adb = shutil.which("adb")
        if which_adb:
            return which_adb

        # Check common Windows BlueStacks 5 & Android SDK paths
        common_windows_paths = [
            r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
            r"C:\Program Files\BlueStacks_nxt\adb.exe",
            r"C:\Program Files (x86)\BlueStacks_nxt\HD-Adb.exe",
            r"C:\Program Files (x86)\BlueStacks_nxt\adb.exe",
            r"C:\Program Files\BlueStacks\HD-Adb.exe",
            r"C:\Program Files\BlueStacks\adb.exe",
            os.path.expanduser(r"~\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
            r"C:\Android\platform-tools\adb.exe",
        ]

        for p in common_windows_paths:
            if os.path.exists(p):
                print(f"[INFO] Auto-detected ADB executable at: '{p}'")
                return p

        # Default fallback (will raise clear error if invoked and not in PATH)
        return "adb"

    def _run_adb(self, args: list, raw: bool = False):
        """Execute an ADB command with detailed error handling."""
        cmd = self._adb_base + args
        try:
            if raw:
                return subprocess.check_output(cmd, stderr=subprocess.PIPE)
            else:
                return subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            raise RuntimeError(
                f"[ERROR] Could not find ADB executable ('{self.adb_exe}').\n"
                "  -> Why: 'adb' or 'HD-Adb.exe' is not in your Windows PATH.\n"
                "  -> How to fix for BlueStacks 5:\n"
                "     1. Pass --adb-path \"C:\\Program Files\\BlueStacks_nxt\\HD-Adb.exe\" when running test_bot, OR\n"
                "     2. Add 'C:\\Program Files\\BlueStacks_nxt\\' to your Windows Environment PATH."
            )
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode("utf-8", errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or "")
            raise RuntimeError(f"ADB command failed: {' '.join(cmd)}\n  -> Error output: {err_msg}")

    def get_screenshot(self) -> np.ndarray:
        """
        Capture a screenshot from the emulator/device and return it as a BGR OpenCV image.
        Uses automatic fallback from exec-out to sdcard file transfer for Windows compatibility.

        :return: np.ndarray of shape (H, W, 3) in BGR format.
        """
        # Method 1: High-speed in-memory pipe ('exec-out screencap -p')
        try:
            raw_png = self._run_adb(["exec-out", "screencap", "-p"], raw=True)
            image_array = np.frombuffer(raw_png, np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is not None and frame.size > 0:
                return frame
        except Exception as e1:
            print(f"[DEBUG] exec-out screencap failed ({e1}). Attempting sdcard file transfer fallback...")

        # Method 2: Fallback to saving on sdcard and pulling (100% reliable on Windows emulators)
        try:
            remote_path = "/sdcard/coc_screencap_temp.png"
            local_path = "coc_screencap_temp.png"
            self._run_adb(["shell", "screencap", "-p", remote_path])
            self._run_adb(["pull", remote_path, local_path])
            self._run_adb(["shell", "rm", remote_path])

            if os.path.exists(local_path):
                frame = cv2.imread(local_path, cv2.IMREAD_COLOR)
                try:
                    os.remove(local_path)
                except Exception:
                    pass
                if frame is not None and frame.size > 0:
                    return frame
        except Exception as e2:
            print(f"[ERROR] Both screencap methods failed. Error: {e2}")

        raise RuntimeError(
            "Could not capture live screen from BlueStacks 5. Please ensure ADB is enabled "
            "in BlueStacks Settings -> Advanced, and check connection with: adb devices"
        )

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
            parts = output.strip().split(":")[-1].strip().split("x")
            return int(parts[0]), int(parts[1])
        except Exception as e:
            print(f"[WARN] Could not query screen resolution ({e}). Defaulting to 1280x720.")
            return 1280, 720
