import pygame
import numpy as np


def draw_arrow(surface, start, end, color, thickness=2):
    if np.linalg.norm(end - start) < 5:
        return
    
    pygame.draw.line(surface, color, start.astype(int), end.astype(int), thickness)
    
    direction = end - start
    length = np.linalg.norm(direction)
    if length > 0:
        direction /= length
        perp = np.array([-direction[1], direction[0]])
        
        arrow_size = min(12, length * 0.3)
        base = end - direction * arrow_size
        left = base + perp * arrow_size * 0.5
        right = base - perp * arrow_size * 0.5
        
        pygame.draw.polygon(surface, color, [
            end.astype(int),
            left.astype(int),
            right.astype(int)
        ])


def draw_goal(surface, position, radius, color, label, font):
    pygame.draw.circle(surface, color, position.astype(int), radius)
    text = font.render(label, True, (255, 255, 255))
    surface.blit(text, (position[0] - 15, position[1] + radius + 5))


def draw_info_panel(surface, font, y_start, info_dict):
    y = y_start
    for key, value in info_dict.items():
        text = font.render(f"{key}: {value}", True, (255, 255, 255))
        surface.blit(text, (10, y))
        y += 25
    return y
