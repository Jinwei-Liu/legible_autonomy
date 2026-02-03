import pygame
import numpy as np
import sys
sys.path.append('..')

from core.legibility import optimize_legible_action, compute_legibility
from utils.visualization import draw_arrow, draw_goal
from config import WIDTH, HEIGHT, BLACK, WHITE, RED, GREEN, BLUE, YELLOW

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dragan Legibility Demo")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)

goals = [
    np.array([650.0, 350.0]),
    np.array([650.0, 400.0]),
    np.array([700.0, 450.0])
]

robot_pos = np.array([100.0, 300.0])
target_idx = 0
legible_mode = True

running = True
while running:
    dt = clock.tick(60) / 1000.0
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                target_idx = 0
            elif event.key == pygame.K_2:
                target_idx = 1
            elif event.key == pygame.K_3:
                target_idx = 2
            elif event.key == pygame.K_SPACE:
                legible_mode = not legible_mode
            elif event.key == pygame.K_r:
                robot_pos = np.array([100.0, 300.0])
    
    if legible_mode:
        action = optimize_legible_action(robot_pos, goals, target_idx)
    else:
        direction = goals[target_idx] - robot_pos
        dist = np.linalg.norm(direction)
        action = 100.0 * direction / dist if dist > 1e-3 else np.zeros(2)
    
    if np.linalg.norm(robot_pos - goals[target_idx]) > 10:
        robot_pos += action * dt
    
    screen.fill(BLACK)
    
    for i, goal in enumerate(goals):
        color = YELLOW if i == target_idx else BLUE
        draw_goal(screen, goal, 15, color, f"G{i+1}", font)
    
    pygame.draw.circle(screen, RED, robot_pos.astype(int), 10)
    
    if np.linalg.norm(action) > 1:
        end = robot_pos + action * 0.5
        draw_arrow(screen, robot_pos, end, GREEN, 3)
    
    mode_text = "LEGIBLE" if legible_mode else "DIRECT"
    info = [
        f"Mode: {mode_text}",
        f"Target: Goal {target_idx + 1}",
        "",
        "1/2/3: Select goal",
        "SPACE: Toggle mode",
        "R: Reset"
    ]
    
    y = 20
    for line in info:
        text = font.render(line, True, WHITE)
        screen.blit(text, (10, y))
        y += 25
    
    pygame.display.flip()

pygame.quit()
