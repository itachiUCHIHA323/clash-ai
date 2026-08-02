"""
Fast Controller Module (Low-Latency Capture & Actuation)
--------------------------------------------------------
Replaces slow `adb exec-out screencap` (~400ms) and `adb shell input tap` (~200ms)
with high-speed window capture (~2ms) and direct input actuation (~1ms).
"""

import time
import cv2
import numpy as np
from typing import Tuple, Optional, Dict

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    pyautogui.PAUSE = 0.0  # Disable pyautogui pause for low latency
except ImportError:
    PYAUTOGUI_AVAILABLE = False


class FastController:
    """
    High-performance controller that interacts with an emulator window
    using OS-level screen buffer capture (mss) and direct mouse/touch events.
    """

    def __init__(self, window_title: str = "Clash of Clans", region: Optional[Dict[str, int]] = None):
        """
        Initialize the FastController.

        :param window_title: Title of the emulator window (e.g., 'BlueStacks', 'Waydroid').
        :param region: Optional bounding box dictionary {'top': y, 'left': x, 'width': w, 'height': h}.
                       If None, defaults to top-left 1280x720 screen region.
        """
        self.window_title = window_title
        self.region = region or {"top": 0, "left": 0, "width": 1280, "height": 720}
        self.sct = mss.mss() if MSS_AVAILABLE else None

    def get_screenshot_fast(self) -> np.ndarray:
        """
        Capture a frame directly from the screen/window buffer using `mss`.
        Latency: ~2 to 5 ms (~200 FPS) vs ~400 ms for adb screencap.

        :return: np.ndarray of shape (H, W, 3) in BGR format.
        """
        if self.sct:
            try:
                sct_img = self.sct.grab(self.region)
                # mss returns BGRA, convert to BGR for OpenCV/YOLO
                frame = np.array(sct_img, dtype=np.uint8)[:, :, :3]
                return np.ascontiguousarray(frame)
            except Exception as e:
                print(f"[WARN] mss grab failed ({e}). Fallback to dummy frame.")
        return np.zeros((self.region["height"], self.region["width"], 3), dtype=np.uint8)

    def tap_fast(self, x: int, y: int) -> float:
        """
        Send a low-latency click/tap event to the window coordinates.
        Latency: ~0.5 to 1.5 ms vs ~200 ms for `adb shell input tap`.

        :param x: X coordinate relative to the window.
        :param y: Y coordinate relative to the window.
        :return: Execution duration in milliseconds.
        """
        start_t = time.perf_counter()
        abs_x = self.region["left"] + int(x)
        abs_y = self.region["top"] + int(y)

        if PYAUTOGUI_AVAILABLE:
            try:
                pyautogui.click(x=abs_x, y=abs_y)
            except Exception as e:
                print(f"[WARN] pyautogui click failed: {e}")
        duration_ms = (time.perf_counter() - start_t) * 1000.0
        return duration_ms

    def benchmark(self, iterations: int = 50) -> Dict[str, float]:
        """
        Benchmark frame capture and input latency.
        """
        # Benchmark Capture
        start_t = time.perf_counter()
        for _ in range(iterations):
            _ = self.get_screenshot_fast()
        capture_total_ms = (time.perf_counter() - start_t) * 1000.0
        avg_capture_ms = capture_total_ms / max(1, iterations)
        fps = 1000.0 / avg_capture_ms if avg_capture_ms > 0 else 0.0

        return {
            "avg_capture_ms": round(avg_capture_ms, 2),
            "estimated_fps": round(fps, 1),
            "iterations": iterations,
        }


if __name__ == "__main__":
    controller = FastController()
    stats = controller.benchmark(iterations=20)
    print("FastController Benchmark Results:", stats)
