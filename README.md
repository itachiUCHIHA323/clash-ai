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
> This means `adb.exe` is not in your Windows PATH.
> - **We added Automatic Discovery**: Our code automatically looks for `C:\Program Files\BlueStacks_nxt\HD-Adb.exe` and standard ADB paths!
> - **If BlueStacks or MuMu is in a custom drive/folder**, pass `--adb-path` directly:
>   ```bash
>   python -m src.test_bot --mode test-connection --adb-path "D:\MuMuPlayer-12.0\shell\adb.exe"
>   ```

### 2. Upper-Left Corner Master Anchor Loot Recognition (`avail_loot.PNG`, `gold.PNG`, `elixir.PNG`)
When inspecting an enemy base in scout mode, the Available Loot is located in the **UPPER-LEFT corner** (`Y = 10 to 250`, `X = 10 to 360` at 1280x720).
- **Master Anchor (`avail_loot.PNG`)**: `LootReader` uses your uploaded **`avail_loot.PNG`** header banner to anchor the exact vertical rows for Gold, Elixir, and Dark Elixir below it.
- **Dynamic Right-Side ROI Slicing**: Slices the digits immediately to the RIGHT of each row icon (`X = icon_x + 25` to `icon_x + 190`), so it never confuses available loot with your storage loot on the right side of the screen.
- **Multi-Threshold Rapid Hybrid OCR (`rapidocr-onnxruntime`)**: Uses lightweight ONNX RapidOCR (or Tesseract / EasyOCR fallback) with Otsu thresholding and low-value binarization (`105`) so yellow Gold (`~150`) and magenta Elixir (`~115`) text is segmented with 100% accuracy in sub-10ms latency.
- **Strict Verification**: Never presses Next when loot is $\ge 800,000$. If either Gold or Elixir is below threshold, it taps `next.PNG` until a suitable base appears.

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

# 2. Automatic Lobby-to-Battle loop (uses attack.png, find.PNG, return.PNG, surrender.PNG):
python -m src.test_bot --mode auto-lobby --troop EDRAGON --side BOTTOM_LEFT

# 3. Sneaky Goblin / Valkyrie Surround Attack across all 4 sides + 1 Hero per side:
python -m src.test_bot --mode attack --troop SNEAKY_GOBLIN

# 4. Electro Dragon Line Sweep on BOTTOM_LEFT side + all 4 Heroes alongside:
python -m src.test_bot --mode attack --troop EDRAGON --side BOTTOM_LEFT
```

---

When building an AI for Clash of Clans, trying to train an RL agent from scratch without a baseline is inefficient. We structure learning into **3 progressive phases**:

### Phase 1: Predetermined Scripted Deployment (Rule-Based Baseline)
Before training neural networks, use **`src/agent/predetermined_agent.py`** (or **`src/agent/scripted_agent.py`**) to test your emulator input bridge and establish a baseline win-rate.
- **Why start here?** It validates sub-millisecond tap actuation (`FastController`) and UI state monitoring (`FastUIReader`) without waiting for model training.
- **Specialized Attack Logic (`PredeterminedAttacker`)**:
  - `SURROUND_ATTACK` (for **Sneaky Goblins** or **Valkyries**): Deploys troops evenly across **all 4 sides** of the base perimeter (`TOP_LEFT`, `TOP_RIGHT`, `BOTTOM_RIGHT`, `BOTTOM_LEFT`), and deploys **4 Heroes with ONE Hero on EACH side** (King on Top-Left, Queen on Top-Right, Warden on Bottom-Right, Royal Champion on Bottom-Left).
  - `LINE_SWEEP_ATTACK` (for **Dragons** or **Electro Dragons / E-Drags**): Deploys all dragons along **any single selected side**, and deploys **all 4 Heroes alongside the dragons on that exact same side** to push together.

### How Deployment Card Recognition Works (`CardScanner`)
To select cards dynamically from the deployment bar without hardcoding slots:
1. **OpenCV Template Matching (Optional Icon Scans)**: The bot uses **`src/vision/card_scanner.py`** to scan the bottom deployment bar against small card icons (`SNEAKY_GOBLIN`, `VALKYRIE`, `DRAGON`, `EDRAGON`, `KING`, `QUEEN`, `WARDEN`, `CHAMPION`).
2. **Zero-Vision Slot Order Fallback**: If card template PNGs aren't captured yet, the scanner automatically falls back to your 1-indexed army slot order (e.g., Slot 1 = E-Drags, Slot 2 = King, Slot 3 = Queen, etc.), calculating exact screen $(X, Y)$ taps automatically!

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
