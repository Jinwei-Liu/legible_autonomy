# Legible Shared Autonomy: Implicit Communication of Robot Belief through Motion

**Albert Author¹ and Bernard D. Researcher²**

¹University of Twente, Netherlands | ²Wright State University, USA

---

## Abstract

Shared autonomy systems combine human input with autonomous assistance to help users with motor impairments control robot arms by inferring user goals and providing guidance. However, the robot's internal beliefs about user goals cannot be observed by users. Traditional shared autonomy systems provide assistance along efficient shortest paths toward inferred goals, but when multiple objects lie in similar directions, such assistive motion remains ambiguous and fails to reveal the specific target identified by the robot. This creates two critical problems. First, when the robot correctly infers the goal, users continue controlling because they cannot perceive understanding from ambiguous assistive motion, wasting effort when autonomous completion would suffice. Second, when the robot misunderstands intent, users cannot quickly detect errors until assistive motion diverges significantly, requiring substantial corrective input. We address this by introducing legible motion into shared autonomy: robot actions must both advance toward the goal and clearly reveal which goal has been inferred, enabling users to understand the robot's beliefs and adjust control accordingly. The robot modulates communication strength through confidence-aware adaptive control authority: providing assertive legible assistance when confident while increasing user authority when uncertain, transforming shared autonomy into transparent bidirectional collaboration. User studies with a six degree-of-freedom robot arm demonstrate X\% reduction in user control effort compared to standard shared autonomy while maintaining task success rates.

---

## Key Contributions

**1. Action-Level Legibility Metric**

We propose the first action-level legibility metric suitable for real-time control:

$$\mathcal{L}(\bm{a}_R \mid s^t, \theta^*) = \log \pi_H(\bm{a}_R \mid s^t, \theta^*) - \log \max_{\theta \neq \theta^*} \pi_H(\bm{a}_R \mid s^t, \theta)$$

Unlike prior trajectory-level approaches that optimize complete paths offline, our metric enables efficient computation at each timestep.

**2. Dual Optimization Framework**

Robot action selection balances legibility and task performance:

$$\bm{a}_R^* = \arg\max_{\bm{a}_R} \left[ \lambda \mathcal{L}(\bm{a}_R \mid s^t, \theta^*) + Q(\bm{a}_R, s^t, \theta^*) \right]$$
---

## Experimental Setup

<p align="center">
  <img src="figures/experimental_setup.pdf" alt="Experimental Setup" width="700"/>
</p>

**Design:** Within-subjects study with 10 participants

**Conditions:**
- λ=0 (Standard SA): Efficient but ambiguous motion
- λ=5 (Medium Legibility): Balanced approach  
- λ=10 (High Legibility): Maximum discriminative motion

**Workspace:** 2D environment (800×600 pixels) with two closely spaced goals creating directional ambiguity

---

## Trajectory Animations

Real participant trajectories provide direct evidence of how legible motion changes collaboration dynamics. Each animation shows all trials for one participant, color-coded by legibility condition (Red: λ=0, Cyan: λ=5, Teal: λ=10).

<p align="center">
  <img src="trajectory_animations/participant_001_trajectories.gif" alt="Participant 001" width="650"/>
  <br>
  <em>Participant 001</em>
</p>

<p align="center">
  <img src="trajectory_animations/participant_002_trajectories.gif" alt="Participant 002" width="650"/>
  <br>
  <em>Participant 002</em>
</p>

<p align="center">
  <img src="trajectory_animations/participant_003_trajectories.gif" alt="Participant 003" width="650"/>
  <br>
  <em>Participant 003</em>
</p>

**Key Observations:**
- Legible conditions (cyan/teal) show earlier divergence toward correct goal
- Standard SA (red) requires more corrections and later disambiguation  
- Users visibly respond to legible motion with reduced control effort

All participant animations are available in `trajectory_animations/participant_*.gif`

---

## Main Results

<p align="center">
  <img src="figures/combined_figure.pdf" alt="Results" width="900"/>
</p>

**Transparency Metrics:**
- Understanding rate: 36% → 94-98% (Standard SA → Legible)
- Prediction accuracy: 48% → 96-98%
- Statistical significance: χ²(2) = 66.01, p < 0.001

**Subjective Experience:**
- Intuitiveness ratings: 3.70 → 7.70 (1-10 scale)
- Collaboration ratings: 3.60 → 7.70
- Strong correlation between measures: r = 0.92, p < 0.001

**Key Finding:** Medium and high legibility produced equivalent benefits, suggesting moderate emphasis suffices for transparency while preserving efficiency.

---

## Repository Structure
```
legible_autonomy/
├── figures/                          # Paper figures
│   ├── experimental_setup.pdf
│   ├── beta_heatmap.pdf
│   └── combined_figure.pdf
│
├── trajectory_animations/            # Participant trajectory GIFs
│   ├── participant_001_trajectories.gif
│   ├── participant_002_trajectories.gif
│   └── ...
│
├── experiment_data/                  # Raw experiment data
├── core/                            # Core algorithms
├── experiment_collection.py         # Main experiment interface
└── analyze_data.py                  # Statistical analysis
```

---

## Reproducing Results

**Installation:**
```bash
git clone https://github.com/Jinwei-Liu/legible_autonomy
cd legible_autonomy
pip install -r requirements.txt
```

**Generate trajectory animations:**
```bash
python generate_trajectory_animations.py --input ./experiment_data --output ./trajectory_animations
```

**Run statistical analysis:**
```bash
python analyze_data.py --input ./experiment_data --output ./figures
```
