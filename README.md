# Clash-AI: Deep Reinforcement Learning for Clash of Clans

`clash-ai` is an open-source framework and architectural blueprint for training an Artificial Intelligence agent to play **Clash of Clans** using **Deep Reinforcement Learning (DRL)** and **Computer Vision**.

---

## 🧠 How Does an AI Learn to Play Clash of Clans?

Unlike board games (Chess, Go) or games with official AI APIs (StarCraft II via `pysc2`), Clash of Clans presents unique challenges:
- **No Official API**: The AI must observe the game visually (screen capture) and interact via simulated touchscreen taps (ADB / Emulator).
- **Complex Tactical Action Space**: Attacking requires selecting a unit/spell card from the deck, choosing an $(X, Y)$ coordinate along the deployment border, and timing the deployment.
- **Delayed & Sparse Rewards**: Destruction percentage and stars update dynamically during a 3-minute battle.

### The 4-Layer Architecture

```
+-------------------------------------------------------------+
|                 4. AI / Learning Engine                     |
|    (Reinforcement Learning: PPO / DQN / Imitation Learning) |
+-------------------------------------------------------------+
               ^ State Grid (Buildings, HP, Troops)
               | Action (Deploy Troop Index @ [X, Y])
+-------------------------------------------------------------+
|              3. Gymnasium RL Environment                    |
|        (Custom Env Wrapper: ClashOfClansEnv)                |
+-------------------------------------------------------------+
     ^ Bounding Boxes, HP Bars     | Simulated Clicks / Taps
     | & Destruction OCR           | (X, Y, Timestamp)
+-------------------------------------------------------------+
|       2. Computer Vision & Perception (Screen -> State)     |
|   (YOLOv8/YOLOv11 + Fast ROI Color/Template UI Matching)    |
+-------------------------------------------------------------+
     ^ Raw Frame Image (RGB)       | Direct Click / Tap Events
     | (mss Window Grab ~2ms)      | (pyautogui / socket ~1ms)
+-------------------------------------------------------------+
|              1. Game Execution / Emulator                   |
|   (Android Emulator: Waydroid, BlueStacks, or Device via ADB)|
+-------------------------------------------------------------+
```

## 🎮 BlueStacks 5 & MuMu Player (`1280x720`) Guide

If you are running **BlueStacks 5 (BST 5)** or **MuMu Player** at **`1280x720`** resolution, our system features automatic emulator discovery:

### 1. Automatic ADB Device Detection (Zero Configuration!)
Whether your emulator appears as **`127.0.0.1:5555`**, **`emulator-5556`** (MuMu Player default), or **`127.0.0.1:7555`**, `ADBController` automatically detects your active connected emulator and switches to it without needing `--adb-serial`!
```bash
# Test connection (automatically detects BlueStacks 5 or MuMu Player):
python -m src.test_bot --mode test-connection
```

> [!TIP]
> **Getting `[WinError 2] The system cannot find the file specified` on Windows?**
> - **We added Local `./adb.exe` & Automatic Discovery**: Our code automatically uses `./adb.exe` in the repository root directory or `C:\Program Files\BlueStacks_nxt\HD-Adb.exe`!
> - **If BlueStacks or MuMu is in a custom drive/folder**, pass `--adb-path` directly:
>   ```bash
>   python -m src.test_bot --mode test-connection --adb-path "D:\MuMuPlayer-12.0\shell\adb.exe"
>   ```

### 2. Upper-Left Corner Master Anchor Loot Recognition (`avail_loot.PNG`, `gold.PNG`, `elixir.PNG`)
When inspecting an enemy base in scout mode, the Available Loot is located in the **UPPER-LEFT corner** (`Y = 10 to 250`, `X = 10 to 360` at 1280x720).
- **Master Anchor (`avail_loot.PNG`)**: `LootReader` uses your uploaded **`avail_loot.PNG`** header banner to anchor the exact vertical rows for Gold, Elixir, and Dark Elixir below it.
- **Zero-Cutoff Right-Side Slicing**: Slices digits starting **2 pixels inside the right edge of each icon** (`X = icon_x + tw - 2` to `+220 px`), so the first digit (`1` in `1,250,000`) is never truncated or missed.
- **Multi-Scale Digit Template Matching (`0.90..1.10` scales)**: To prevent emulator DPI differences from adding extra digits or skipping digits, `_match_digit_templates` scales each `0.png..9.png` template across 5 DPI scales and applies horizontal Non-Maximum Suppression (NMS) for 100% precision.
- **Multi-Mode Rapid Hybrid OCR (`rapidocr-onnxruntime`) as Engine #2**: Tests Bright-Pixel Binary (`140`), Otsu, and Raw BGR across RapidOCR / EasyOCR / Tesseract if template matching confidence varies.
- **Interactive Loot Diagnostic (`test-loot`)**:
  ```bash
  python -m src.test_bot --mode test-loot
  ```
  Captures a live screen, prints extracted Gold/Elixir/Dark Elixir, and saves annotated debug images (`debug_loot_screen_boxes.png`, `debug_loot_gold_roi.png`).
- **Strict Verification**: Never presses Next when loot is $\ge 800,000$. (Use `--force-attack` to bypass loot check for manual testing).

### 3. Real-Scanning Troop Count Badges (`read_troop_count`) & Dynamic Army Discovery (`scan_available_army`)
Before deploying an attack, `CardScanner.scan_available_army(frame)` scans your entire bottom deployment bar:
- **True Troop Count Recognition**: Crops the number badge directly above each card (`Y = card_y - 30 to card_y - 6`, `X = card_x - 16 to card_x + 16`) and reads the exact remaining troop count (e.g., `28` Valkyries, `8` Dragons).
- **Dynamic Outermost Edge Deployment**: Deploys only cards that are actually available in your bar (`count > 0`), automatically dividing the real unit count across the 4 sides for `SURROUND_ATTACK` or sweeping along 1 side for `LINE_SWEEP_ATTACK`.

### 3. Integrated Asset & Template Library (From `CoC-ai-gud`)
We imported all PNG image assets from `itachiUCHIHA323/CoC-ai-gud` into `templates/ui/` and `templates/cards/`:
- **UI & Battle Buttons**: Over 50 lobby, matchmaking, dialog, and battle buttons (`attack.png`, `confirm_attack.png`, `find_match.png`, `next.png`, `return_home.png`, `sur_end.png`, etc.).
- **Troop & Hero Cards**: Over 40 unit and spell icons (`valkyrie.png`, `super_valkyrie.png`, `sneaky_goblin.png`, `E_dragon.png`, `dragon.png`, `king.png`, `queen.png`, `wd.png`, `rc.png`, etc.).
*(Note: `CoC-ai-gud` contains only compiled `.exe`/`.so` binaries without source code, so modifying password authentication in those closed-source binaries is not feasible or necessary — our open-source Python bot in `clash-ai` is fully inspectable, password-free, and under your complete control!)*

### 2. Print 1280x720 Coordinate Table & Generate Calibration Overlay
To inspect exact pixel `(X, Y)` tap locations for all 8 card slots and the 4 base sides:
```bash
python -m src.test_bot --mode calibrate --adb-serial 127.0.0.1:5555
```
This outputs a clean ASCII coordinate table and saves **`debug_calibration_1280x720.png`** showing every tap point overlayed on your screen.

#### Exact 1280x720 Pixel Coordinate Reference Table:
- **Deployment Card Slots (`Y = 655 px`)**:
  - `Slot 1 (Troop)`: `X = 153`
  - `Slot 2 (King)`: `X = 262`
  - `Slot 3 (Queen)`: `X = 371`
  - `Slot 4 (Warden)`: `X = 480`
  - `Slot 5 (Royal Champ)`: `X = 588`
  - `Slot 6-8 (Spells/Support)`: `X = 697, 806, 915`
- **4 Base Perimeter Sides (`Start -> Midpoint (Hero Drop) -> End`)**:
  - `TOP_LEFT` Side: `(230, 324) -> (390, 234) -> (550, 144)`
  - `TOP_RIGHT` Side: `(730, 144) -> (890, 234) -> (1050, 324)`
  - `BOTTOM_RIGHT` Side: `(1050, 396) -> (890, 486) -> (730, 576)`
  - `BOTTOM_LEFT` Side: `(550, 576) -> (390, 486) -> (230, 396)`

### 3. Launch an Attack on BlueStacks 5
Run a predetermined attack directly from the command line:
```bash
# 1. State-Verified Continuous Farming Loop (800k loot -> Pinch-Out -> Outermost Edge Deploy -> Return Home):
python -m src.auto_farmer --troop VALKYRIE --min-gold 800000 --min-elixir 800000 --cycles 5

# 2. 4-Step Tactical Attack Demo on Full-Base Screenshots (Generates visual attack overlays):
python -m src.test_bot --mode test-attack-demo

# 3. Automatic Lobby-to-Battle loop (uses attack.png, find.PNG, return.PNG, surrender.PNG):
python -m src.test_bot --mode auto-lobby --troop EDRAGON --side BOTTOM_LEFT

# 4. Sneaky Goblin / Valkyrie Surround Attack across all 4 sides + 1 Hero per side:
python -m src.test_bot --mode attack --troop SNEAKY_GOBLIN

# 5. Electro Dragon Line Sweep on BOTTOM_LEFT side + all 4 Heroes alongside:
python -m src.test_bot --mode attack --troop EDRAGON --side BOTTOM_LEFT
```

---

When building an AI for Clash of Clans, trying to train an RL agent from scratch without a baseline is inefficient. We structure learning into **3 progressive phases**:

### Phase 1: Predetermined Scripted Deployment (4-Step Tactical Baseline)
Before training neural networks, use **`src/agent/predetermined_agent.py`** to execute specialized 4-Step Tactical Attacks:
1. **STEP 1 (Troops First - ALL OF THEM UNTIL GREYED OUT)**: Selects troop cards and taps along the absolute outermost green grass border (`margin = 11%`, `0.12 to 0.88`) **until the card icon turns greyed out (mean Saturation `< 40`)**, guaranteeing every single unit is dropped without "cannot deploy here" red zone errors.
2. **STEP 2 (Then Heroes)**: Deploys all available Heroes (`KING`, `QUEEN`, `WARDEN`, `CHAMPION`) *after* all troops are greyed out (1 per side for Surround, alongside troops for Line Sweep).
3. **STEP 3 (Spells A BIT AHEAD - NOT AT BACK)**: Deploys Rage and Support spells slightly INWARD toward the center of the base (`~18%` ahead of the troop drop line), placing the spell circle right where troops walk into enemy defenses.
4. **STEP 4 (Hero Ability Activation)**: Waits `~6.5 seconds` after heroes engage defenses and taps all Hero cards in the deployment bar again to trigger Hero Abilities (Gauntlet/Tome/Arrow).

### How Deployment Card Recognition Works (`CardScanner`)
To select cards dynamically from the deployment bar without hardcoding slots:
1. **Competitive Best-Match per Slot (`threshold >= 0.75`)**: Evaluates all card templates per slot along the bottom bar (`Y = 76% to 98%`) and assigns the highest-scoring card, preventing false matches (e.g., matching Valkyries when Dragons are present).
2. **True Troop Count Recognition (`read_troop_count`) & `scan_available_army`**: Crops the number badge above each card icon (`Y = card_y - 32 to card_y - 5`, `X = card_x - 18 to card_x + 18`) and reads the exact remaining count using `clash2`'s `battleTroopCountFont/0.png..9.png`.
3. **Greyed-Out Card Detection (`is_card_greyed_out`)**: Converts card icon ROI to HSV and checks mean Saturation (`< 40`). Active cards have saturation `> 55`; deployed empty cards become desaturated grey.

### What Images / Assets Do You Need?
- **Do we need images for the 4 sides of the base?** **No!** A Clash of Clans battle map is an invariant isometric diamond. The 4 perimeter edges (`TOP_LEFT`, `TOP_RIGHT`, `BOTTOM_RIGHT`, `BOTTOM_LEFT`) are mapped geometrically in normalized coordinates (`0.0 - 1.0`) inside `PredeterminedAttacker`.
- **Do we need images for Attack / End Battle buttons?** Only small button templates (`templates/ui/attack_btn.png`, `end_battle_btn.png`) or fixed screen UI percentages if automating the full lobby search loop.

### Phase 2: Behavioral Cloning (Imitation Learning Bootstrap)
Instead of letting an RL agent drop spells randomly on empty grass:
1. Log coordinate tuples `(timestamp, card_idx, x, y)` from your Phase 1 scripted attacks (or human player replays).
2. Train a supervised neural network to predict the next deployment action from the game screen.
3. This bootstraps the policy network so it starts with intermediate tactical competence.

### Phase 3: Proximal Policy Optimization (PPO Reinforcement Learning)
Once initialized from Phase 1 & 2, the agent trains inside **`ClashOfClansEnv`** using Stable-Baselines3 PPO:
- Through trial and error across thousands of automated attacks, the model optimizes spell timings, funneling angles, and hero ability activations to maximize 3-star win rates.

---

## 🔬 Core Learning Methodologies

### 1. Proximal Policy Optimization (PPO) - Reinforcement Learning
We model the attacking phase as a **Markov Decision Process (MDP)**:
- **Observation State ($s_t$)**: A multi-channel grid representing:
  - Building locations and remaining HP.
  - Defensive coverage/threat zones (Inferno Tower, Eagle Artillery, Air Defenses).
  - Valid deployment zone mask (outer border where troops can be dropped).
  - Remaining troops and spells in the deployment bar.
- **Action ($a_t$)**: A tuple $(c, x, y)$ where:
  - $c \in [0, K]$ is the selected troop/spell card index (or $0$ for "Wait/No-op").
  - $(x, y)$ is the normalized coordinate on the battle map.
- **Reward Function ($r_t$)**:
  - $+1.0$ for each Star earned (Town Hall destroyed, 50% destruction, 100% destruction).
  - $+0.01 \times \Delta(\text{Destruction \%})$ continuous progress reward.
  - $-0.05$ penalty for invalid taps (e.g., trying to drop troops inside red restricted zones or when out of troops).

### 2. Behavioral Cloning (Imitation Learning Bootstrap)
To prevent the agent from spending thousands of episodes dropping troops randomly:
1. Record gameplay videos and tap logs of skilled human players.
2. Train a supervised policy network to predict human taps $(x, y, \text{troop})$ from screenshots.
3. Fine-tune the pre-trained model using PPO so the agent can discover superhuman strategies.

---

## 🚀 Repository Structure

```
clash-ai/
├── README.md                  # This document
├── requirements.txt           # Python dependencies
├── templates/                 # Image template directories (cards & UI buttons)
├── src/
│   ├── auto_farmer.py         # Master Continuous Auto-Farming Loop (Lobby -> Loot OCR -> Edge Deploy -> Return Home)
│   ├── test_bot.py            # Master CLI runner for BlueStacks 5 testing, calibration & attacks
│   ├── test_loot.py           # Interactive Loot OCR diagnostic & visual bounding box calibrator
│   ├── test_attack_on_demo.py # 4-Step Tactical Attack Demo & Calibration on User Full-Base Screenshots
│   ├── calibrate_coords.py    # 1280x720 coordinate table generator & debug overlay creator
│   ├── controller/
│   │   ├── fast_controller.py # Ultra-low latency mss window capture & pyautogui taps (~3ms)
│   │   └── adb_controller.py  # Standard ADB emulator screen capture & tap fallback
│   ├── vision/
│   │   ├── ui_matcher.py      # Auto-navigation using user's attack.png, find.PNG, surrender.PNG
│   │   ├── loot_reader.py     # Gold & Elixir OCR scanner & threshold comparator (800,000 default)
│   │   ├── card_scanner.py    # Deployment bar card recognition (Templates + Slot Order fallback)
│   │   ├── fast_ocr.py        # Sub-millisecond ROI template matching & HSV star detector
│   │   └── detector.py        # YOLOv8 + OpenCV perception pipeline
│   ├── env/
│   │   └── clash_env.py       # Custom Gymnasium environment (ClashOfClansEnv)
│   └── agent/
│       ├── predetermined_agent.py # Surround (Valk/Sneaky Gob) & Line Sweep (Drag/E-Drag) + 4 Heroes
│       ├── scripted_agent.py  # Phase 1: Predetermined rule-based strategies (BARCH, Funnel, Surround)
│       ├── train_ppo.py       # Training script using Stable-Baselines3 PPO
│       └── random_agent.py    # Baseline agent testing random deployment
```

---

## 🛠️ Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Your Emulator / Device
- Enable **USB Debugging** or **ADB Network Debugging** on your Android emulator (e.g., BlueStacks, Waydroid, MuMu, or physical device).
- Verify ADB connection:
```bash
adb devices
```

### 3. Test the ADB & Vision Pipeline
```bash
python -m src.vision.detector
```

### 4. Run a Baseline Predetermined or Scripted Agent
```bash
# Test Sneaky Goblin / Valkyrie 4-side surround or E-Dragon line sweep + 4 Heroes
python -m src.agent.predetermined_agent

# Test predetermined BARCH Wave or Giant+Wizard funneling strategy
python -m src.agent.scripted_agent
```

### 5. Train the PPO Attacking Agent
```bash
python -m src.agent.train_ppo --total-timesteps 100000
```
