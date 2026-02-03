WIDTH, HEIGHT = 800, 600

BLACK, WHITE = (0, 0, 0), (255, 255, 255)
RED, GREEN = (255, 0, 0), (0, 255, 0)
BLUE, YELLOW, CYAN = (100, 100, 255), (255, 255, 0), (0, 255, 255)

# Speed parameters
ROBOT_SPEED = 50.0     # Robot maximum velocity
USER_SPEED = 50.0       # User input speed scale

# Time step for dynamics (seconds)
DT = 0.1  # Time step used in Q(aR, st, θ*) = -||θ* - (st + Δt·aR)||

# Human action model parameters (Equations 3-4)
BETA_RATIONALITY = 0.1  # βr: Rationality parameter in πH(a|s,θ)
EFFORT_WEIGHT = 1    # λe: Effort penalty in R(a,s,θ)

# Adaptive authority allocation parameters (Equations 10-12)
BETA_BASE = 0.6         # βbase: Baseline user authority
BETA_MIN = 0.2          # βmin: Minimum user authority (max robot control)
B_THRESH = 0.6          # bthresh: Confidence threshold for robot control
D_MIN = 0.1             # dmin: Distance threshold (fraction of workspace)
D_MAX = 0.4             # dmax: Distance threshold (fraction of workspace)

# Optimization parameters (Equation 8)
# Balance between legibility and task performance: score = L + λ·Q
TASK_WEIGHT = 0.3       # λ: Scaling for task performance (for demo)
                         # Chosen to match magnitude: λ ≈ typical(|L|)/typical(|Q|) ≈ 3/200
SEARCH_ANGLES = 11       # Number of candidate actions to search
ANGLE_RANGE = 1.0        # Search range in radians (±60 degrees ≈ 1.047)

# Experiment parameters
TASK_WEIGHT_LIST = [0.1, 0.3, 0.5]  # Different λ values to test
TRIALS_PER_CONDITION = 5  # Number of trials per TASK_WEIGHT value
MIDPOINT_THRESHOLD = 0.5  # When to show questionnaire (0.5 = halfway)

# Visualization
GOAL_RADIUS = 20
ROBOT_RADIUS = 10
FPS = 30
