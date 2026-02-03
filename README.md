# Legible Shared Autonomy

Implementation of legible shared autonomy where robots communicate their understanding of user intent through motion.

## Structure

```
legible_autonomy/
├── core/
│   ├── legibility.py          # Legibility metrics
│   └── shared_autonomy.py     # Shared autonomy with legible communication
├── utils/
│   └── visualization.py       # Visualization utilities
├── config.py                  # Configuration parameters
├── demo_shared.py             # Shared autonomy demo
└── demo_dragan.py             # Action-level legibility demo (validation)
```

## Installation

```bash
pip install -r requirements.txt
```

## Demos

### 1. Shared Autonomy Demo

```bash
python demo_shared.py
```

**Controls:**
- Arrow keys: Control robot
- R: Reset

**Observe:**
- White arrow: User input
- Green arrow: Robot action (exaggerated when confident and close to goal)
- Yellow arrow: Executed action
- β value: Blending parameter (adaptive based on confidence and distance)
- Belief bars: Robot's belief about user's goal

### 2. Dragan Legibility Demo

Validates that action-level legibility produces similar behavior to trajectory-level optimization.

```bash
python demo_dragan.py
```

**Controls:**
- 1/2/3: Select target goal
- SPACE: Toggle legible/direct motion
- R: Reset

**Observe:**
- Green arrow: Robot action
- Legible mode: Action optimized for max legibility + task performance
- Direct mode: Direct path to goal

## Method

Robot optimizes its action to balance:
1. **Legibility**: L(a_R) = log P(θ*|a_R) - log max P(θ|a_R)
2. **Task performance**: Q(a_R) = -||θ* - (s + a_R)||

**Action optimization:**
```
a_R* = argmax [λ·L(a_R) + (1-λ)·Q(a_R)]
```

**Adaptive authority allocation:**
```
β^t = β_base - α(b_max) · γ(d) · (β_base - β_min)
```
where α increases with confidence, γ increases as robot approaches goal.

**Execution:**
- User releases control: execute a_R directly
- User provides input: blend as β·a_H + (1-β)·a_R

## Parameters (config.py)

- `LAMBDA_LEG = 0.5`: Legibility vs task trade-off
- `BETA_BASE = 0.6`: Baseline user authority
- `BETA_MIN = 0.2`: Minimum user authority
- `B_THRESH = 0.6`: Confidence threshold for adaptive authority
- `D_MIN/D_MAX = 0.1/0.4`: Distance thresholds (fraction of workspace)
- `BETA_RATIONALITY = 1.5`: Human rationality (lower = more gradual belief updates)
- `EFFORT_WEIGHT = 0.01`: Control effort penalty

**Note:** Goals should be at least 150-250 pixels apart for smooth belief updates.
