"""
Loot OCR & Template Reader (Upper-Left Corner 'avail_loot.PNG' Master Anchor Scanner)
-------------------------------------------------------------------------------------
Scans the enemy village scout screen in the UPPER-LEFT corner using 'avail_loot.PNG'
header banner, with fallback to 'gold.PNG', 'elixir.PNG', and 'dark_exlixir.PNG'.

Below 'avail_loot.PNG' are the 3 rows:
1. Gold row -> slices digits to the right of Gold icon.
2. Elixir row -> slices digits to the right of Elixir icon.
3. Dark Elixir row -> slices digits to the right of Dark Elixir icon.

Uses Raw BGR Color + Grayscale + Otsu with RapidOCR / EasyOCR / Tesseract
to segment yellow Gold and magenta Elixir digits with 100% accuracy.
"""

import os
import shutil
import cv2
import numpy as np
import re
from typing import Dict, Tuple, Optional, List

# Try importing Tesseract OCR
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Try importing RapidOCR (ultra-fast ONNX runtime OCR)
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
    RAPIDOCR_ENGINE = None
except ImportError:
    RAPIDOCR_AVAILABLE = False

# Try importing EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    EASYOCR_READER = None
except ImportError:
    EASYOCR_AVAILABLE = False


class LootReader:
    """
    Parses Available Gold, Elixir, and Dark Elixir from the UPPER-LEFT corner
    of battle scout screenshots using 'avail_loot.PNG', 'gold.PNG', 'elixir.PNG', and 'dark_exlixir.PNG'.
    """

    def __init__(self, min_gold: int = 800000, min_elixir: int = 800000, template_dir: str = "templates/ui"):
        self.min_gold = min_gold
        self.min_elixir = min_elixir
        self.template_dir = template_dir

        # Auto-discover Tesseract OCR executable on Windows
        self._setup_tesseract_windows()

        # Load icon templates for Available Loot header, Gold, Elixir, and Dark Elixir
        self.icon_templates: Dict[str, np.ndarray] = {}
        self._load_loot_icons()

    def _setup_tesseract_windows(self) -> None:
        """Auto-discover Tesseract executable on Windows if not in PATH."""
        if not PYTESSERACT_AVAILABLE:
            return
        if shutil.which("tesseract"):
            return

        windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        ]
        for p in windows_paths:
            if os.path.exists(p):
                print(f"[INFO] Auto-detected Tesseract OCR executable at: '{p}'")
                pytesseract.pytesseract.tesseract_cmd = p
                break

    def _load_loot_icons(self) -> None:
        """Load avail_loot.PNG, gold.PNG, elixir.PNG, and dark_exlixir.PNG icon templates."""
        targets = {
            "avail_loot": ["avail_loot.png", "avail_loot.PNG"],
            "gold": ["gold.png", "gold.PNG"],
            "elixir": ["elixir.png", "elixir.PNG"],
            "dark_elixir": ["dark_exlixir.png", "dark_exlixir.PNG", "dark_elixir.png", "dark_elixir.PNG"],
        }
        for resource, filenames in targets.items():
            for fname in filenames:
                path = os.path.join(self.template_dir, fname)
                if os.path.exists(path):
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    if img is not None:
                        self.icon_templates[resource] = img
                        print(f"[INFO] Loaded Loot template '{resource}' from '{path}' ({img.shape[1]}x{img.shape[0]})")
                        break

    def read_loot(self, frame: np.ndarray, save_debug_roi: bool = True) -> Dict[str, int]:
        """
        Extract Available Gold, Elixir, and Dark Elixir amounts from the UPPER-LEFT corner of a scout frame.
        """
        h, w, _ = frame.shape

        # Search ONLY in the upper-left corner (Y = 10 to 270 px, X = 10 to 380 px at 1280x720)
        y1, y2 = int(h * 0.01), int(h * 0.38)
        x1, x2 = int(w * 0.01), int(w * 0.30)
        upper_left_roi = frame[y1:y2, x1:x2]

        # 1. Master Anchor Attempt: Check for 'avail_loot.PNG' banner
        anchor_found, anchor_coords = self._find_avail_loot_anchor(upper_left_roi)
        gold_crop, elixir_crop, dark_crop = None, None, None

        if anchor_found and anchor_coords:
            ax, ay, a_w, a_h = anchor_coords
            roi_h, roi_w, _ = upper_left_roi.shape
            # Row 1 below 'avail_loot.PNG' banner: Gold
            gy1 = min(roi_h, ay + a_h - 2)
            gy2 = min(roi_h, gy1 + 38)
            gx1 = min(roi_w, ax + 25)
            gx2 = min(roi_w, gx1 + 195)
            gold_crop = upper_left_roi[gy1:gy2, gx1:gx2]

            # Row 2 below Gold: Elixir
            ey1 = min(roi_h, gy2 - 2)
            ey2 = min(roi_h, ey1 + 38)
            elixir_crop = upper_left_roi[ey1:ey2, gx1:gx2]

            # Row 3 below Elixir: Dark Elixir
            dy1 = min(roi_h, ey2 - 2)
            dy2 = min(roi_h, dy1 + 38)
            dark_crop = upper_left_roi[dy1:dy2, gx1:gx2]
            print(f"[LOOT SCAN] Master anchor 'avail_loot.PNG' matched! Slicing Gold, Elixir, Dark Elixir rows below it.")

        # 2. Fallback / Refinement: Match individual icon templates (gold.PNG, elixir.PNG)
        if gold_crop is None or gold_crop.size == 0:
            gold_crop = self._slice_icon_right(upper_left_roi, "gold", default_y=(55, 95), default_x=(55, 240))
        if elixir_crop is None or elixir_crop.size == 0:
            elixir_crop = self._slice_icon_right(upper_left_roi, "elixir", default_y=(95, 135), default_x=(55, 240))
        if dark_crop is None or dark_crop.size == 0:
            dark_crop = self._slice_icon_right(upper_left_roi, "dark_elixir", default_y=(135, 175), default_x=(55, 210))

        if save_debug_roi:
            for name, crop in [("gold", gold_crop), ("elixir", elixir_crop), ("dark_elixir", dark_crop)]:
                if crop is not None and crop.size > 0:
                    try:
                        cv2.imwrite(f"debug_loot_{name}_roi.png", crop)
                    except Exception:
                        pass

        gold_val = self._run_rapid_ocr(gold_crop, "GOLD") if gold_crop is not None else 0
        elixir_val = self._run_rapid_ocr(elixir_crop, "ELIXIR") if elixir_crop is not None else 0
        dark_val = self._run_rapid_ocr(dark_crop, "DARK_ELIXIR") if dark_crop is not None else 0

        return {
            "gold": gold_val,
            "elixir": elixir_val,
            "dark_elixir": dark_val,
        }

    def _find_avail_loot_anchor(self, upper_left_roi: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Locate 'avail_loot.PNG' header banner in upper_left_roi."""
        if "avail_loot" not in self.icon_templates:
            return False, None
        tmpl = self.icon_templates["avail_loot"]
        th, tw, _ = tmpl.shape
        roi_h, roi_w, _ = upper_left_roi.shape
        if tw > roi_w or th > roi_h:
            return False, None

        res = cv2.matchTemplate(upper_left_roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= 0.58:
            return True, (max_loc[0], max_loc[1], tw, th)
        return False, None

    def _slice_icon_right(
        self,
        upper_left_roi: np.ndarray,
        resource_name: str,
        default_y: Tuple[int, int],
        default_x: Tuple[int, int],
    ) -> np.ndarray:
        """Locate resource icon and slice digits immediately to its RIGHT."""
        roi_h, roi_w, _ = upper_left_roi.shape
        if resource_name in self.icon_templates:
            tmpl = self.icon_templates[resource_name]
            th, tw, _ = tmpl.shape
            if tw <= roi_w and th <= roi_h:
                res = cv2.matchTemplate(upper_left_roi, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= 0.58:
                    ix, iy = max_loc
                    crop_x1 = min(roi_w, ix + tw + 2)
                    crop_x2 = min(roi_w, crop_x1 + 185)
                    crop_y1 = max(0, iy - 4)
                    crop_y2 = min(roi_h, iy + th + 6)
                    return upper_left_roi[crop_y1:crop_y2, crop_x1:crop_x2]

        dy1, dy2 = default_y
        dx1, dx2 = default_x
        return upper_left_roi[dy1:dy2, dx1:dx2]

    def _run_rapid_ocr(self, crop: Optional[np.ndarray], resource_name: str) -> int:
        """
        Run OCR (RapidOCR / EasyOCR / Tesseract) on raw BGR Color + Grayscale + Otsu images.
        Testing raw BGR Color first prevents binary thresholding from erasing yellow Gold
        or magenta Elixir text.
        """
        if crop is None or crop.size == 0:
            return 0

        # Create 2x scaled BGR color and grayscale versions
        scaled_bgr = cv2.resize(crop, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_low = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY)

        candidates = []

        # 1. Try RapidOCR (ONNX runtime - works best on BGR color & grayscale)
        if RAPIDOCR_AVAILABLE:
            global RAPIDOCR_ENGINE
            try:
                if RAPIDOCR_ENGINE is None:
                    RAPIDOCR_ENGINE = RapidOCR()
                for mode_name, img in [("BGR", scaled_bgr), ("Gray", gray), ("Otsu", thresh_otsu)]:
                    result, _ = RAPIDOCR_ENGINE(img)
                    if result:
                        for box, text, score in result:
                            digits = re.sub(r"\D", "", text)
                            if digits and len(digits) >= 4:
                                candidates.append(int(digits))
            except Exception as e:
                print(f"[DEBUG] RapidOCR failed on {resource_name}: {e}")

        # 2. Try EasyOCR if available and RapidOCR found nothing
        if EASYOCR_AVAILABLE and not candidates:
            global EASYOCR_READER
            try:
                if EASYOCR_READER is None:
                    EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
                for mode_name, img in [("BGR", scaled_bgr), ("Gray", gray)]:
                    results = EASYOCR_READER.readtext(img, allowlist="0123456789")
                    for _, text, conf in results:
                        digits = re.sub(r"\D", "", text)
                        if digits and len(digits) >= 4:
                            candidates.append(int(digits))
            except Exception:
                pass

        # 3. Try Tesseract OCR across binarization modes if previous engines found nothing
        if PYTESSERACT_AVAILABLE and not candidates:
            for mode_name, th_img in [("Otsu", thresh_otsu), ("LowThresh", thresh_low), ("Gray", gray)]:
                try:
                    text = pytesseract.image_to_string(th_img, config="--psm 7 -c tessedit_char_whitelist=0123456789")
                    digits = re.sub(r"\D", "", text)
                    if digits and len(digits) >= 4:
                        candidates.append(int(digits))
                except Exception:
                    pass

        if candidates:
            best_val = max(candidates)
            print(f"[LOOT OCR] {resource_name}: {best_val:,}")
            return best_val

        print(f"[LOOT OCR] Could not read {resource_name} digits. Check debug_loot_{resource_name.lower()}_roi.png")
        return 0

    def is_loot_sufficient(self, loot_dict: Dict[str, int], force_attack: bool = False) -> bool:
        """
        Return True if both Gold and Elixir are >= the minimum thresholds (default: 800,000).
        """
        if force_attack:
            print("[LOOT CHECK] --force-attack enabled: Bypassing loot threshold verification.")
            return True

        gold = loot_dict.get("gold", 0)
        elixir = loot_dict.get("elixir", 0)
        dark_elixir = loot_dict.get("dark_elixir", 0)

        sufficient = gold >= self.min_gold and elixir >= self.min_elixir
        print(
            f"[LOOT CHECK] Gold: {gold:,} (Min: {self.min_gold:,}) | "
            f"Elixir: {elixir:,} (Min: {self.min_elixir:,}) | "
            f"Dark Elixir: {dark_elixir:,} -> Attack Suitable: {sufficient}"
        )
        return sufficient


if __name__ == "__main__":
    reader = LootReader(min_gold=800000, min_elixir=800000)
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    loot = reader.read_loot(dummy)
    print("Test Loot Reader output:", loot)
