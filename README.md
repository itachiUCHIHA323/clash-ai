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
├── src/
│   ├── controller/
│   │   ├── fast_controller.py # Ultra-low latency mss window capture & pyautogui taps (~3ms)
│   │   └── adb_controller.py  # Standard ADB emulator screen capture & tap fallback
│   ├── vision/
│   │   ├── fast_ocr.py        # Sub-millisecond ROI template matching & HSV star detector
│   │   └── detector.py        # YOLOv8 + OpenCV perception pipeline
│   ├── env/
│   │   └── clash_env.py       # Custom Gymnasium environment (ClashOfClansEnv)
│   └── agent/
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

### 4. Run a Baseline Random Agent
```bash
python -m src.agent.random_agent
```

### 5. Train the PPO Attacking Agent
```bash
python -m src.agent.train_ppo --total-timesteps 100000
```
