import pygame
import numpy as np
import sys
sys.path.append('..')

from core.shared_autonomy import LegibleSharedAutonomy
from utils.visualization import draw_arrow, draw_goal
from config import WIDTH, HEIGHT, BLACK, WHITE, RED, GREEN, BLUE, YELLOW, CYAN, USER_SPEED

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Legible Shared Autonomy Demo - PS5 Controller")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
font_large = pygame.font.Font(None, 48)

# Initialize joystick (PS5 controller)
pygame.joystick.init()
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Controller connected: {joystick.get_name()}")
else:
    print("Warning: No controller detected! Please connect PS5 controller.")

# Goals setup
goals = [
    np.array([600.0, 380.0]),
    np.array([600.0, 420.0])
]
robot_pos = np.array([100.0, 400.0])
sa = LegibleSharedAutonomy(goals)

# Dead zone for joystick
DEAD_ZONE = 0.15

running = True
while running:
    dt = clock.tick(60) / 1000.0
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Reset with Circle button
        elif event.type == pygame.JOYBUTTONDOWN:
            if event.button == 1:
                robot_pos = np.array([100.0, 400.0])
                sa.beliefs = np.ones(len(goals)) / len(goals)
                print("Reset position and beliefs")
    
    # Get user input from PS5 controller left joystick
    user_input = np.array([0.0, 0.0])
    if joystick:
        # Left stick: axis 0 = horizontal, axis 1 = vertical
        x_axis = joystick.get_axis(0)
        y_axis = joystick.get_axis(1)
        
        # Apply dead zone
        if abs(x_axis) > DEAD_ZONE:
            user_input[0] = x_axis * USER_SPEED
        if abs(y_axis) > DEAD_ZONE:
            user_input[1] = y_axis * USER_SPEED
    
    # Update belief and compute robot action
    sa.update_belief(robot_pos, user_input)
    robot_action = sa.compute_robot_action(robot_pos, user_input)
    
    # Execute blended action
    executed = robot_action if np.linalg.norm(user_input) < 0.01 else \
               sa.beta * user_input + (1 - sa.beta) * robot_action
    
    robot_pos = np.clip(robot_pos + executed * dt, [0, 0], [WIDTH, HEIGHT])
    
    # Rendering
    screen.fill(BLACK)
    
    # Draw goals with highlighting for max belief
    max_belief_idx = np.argmax(sa.beliefs)
    for i, goal in enumerate(goals):
        if i == max_belief_idx:
            # Highlight the goal with highest belief
            # Draw glowing effect
            for radius in [25, 20, 15]:
                alpha = 100 - (25 - radius) * 3
                glow_color = (0, 255, 0, alpha)  # Green glow
                s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, glow_color, (radius, radius), radius)
                screen.blit(s, (int(goal[0] - radius), int(goal[1] - radius)))
            
            # Draw highlighted goal
            draw_goal(screen, goal, 15, YELLOW, f"G{i}", font)
            # Add pulsing border
            pygame.draw.circle(screen, CYAN, goal.astype(int), 20, 3)
        else:
            draw_goal(screen, goal, 15, GREEN, f"G{i}", font)
    
    # Draw robot
    pygame.draw.circle(screen, RED, robot_pos.astype(int), 10)
    
    # Draw arrows
    scale = 0.5
    if np.linalg.norm(user_input) > 1:
        draw_arrow(screen, robot_pos, robot_pos + user_input * scale, WHITE, 2)
    if np.linalg.norm(robot_action) > 1:
        draw_arrow(screen, robot_pos, robot_pos + robot_action * scale, GREEN, 3)
    if np.linalg.norm(executed) > 1:
        draw_arrow(screen, robot_pos, robot_pos + executed * scale, YELLOW, 2)
    
    # Display info
    y = 20
    beta_val = sa.beta
    if abs(beta_val - 0.6) > 0.05:
        text = font_large.render(f"β={beta_val:.2f}", True, CYAN)
        screen.blit(text, (10, y))
        y += 60
    else:
        text = font.render(f"β={beta_val:.3f}", True, (100, 100, 100))
        screen.blit(text, (10, y))
        y += 30
    
    text = font.render("Beliefs:", True, WHITE)
    screen.blit(text, (10, y))
    y += 25
    
    # Draw belief bars with highlighting
    for i, b in enumerate(sa.beliefs):
        # Highlight the max belief bar
        if i == max_belief_idx:
            color = YELLOW
            border_color = CYAN
            border_width = 2
        else:
            color = GREEN if i == 0 else BLUE
            border_color = WHITE
            border_width = 1
        
        bar_w = int(b * 200)
        pygame.draw.rect(screen, color, (10, y, bar_w, 15))
        pygame.draw.rect(screen, border_color, (10, y, 200, 15), border_width)
        
        text_color = CYAN if i == max_belief_idx else WHITE
        text = font.render(f"{b:.2f}", True, text_color)
        screen.blit(text, (220, y))
        y += 20
    
    # Controller instructions
    y = HEIGHT - 60
    controller_info = [
        "Left Stick: Control",
        "○: Reset"
    ]
    for line in controller_info:
        text = font.render(line, True, (150, 150, 150))
        screen.blit(text, (10, y))
        y += 20
    
    pygame.display.flip()

if joystick:
    joystick.quit()
pygame.quit()
