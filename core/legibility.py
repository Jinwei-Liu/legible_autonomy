import numpy as np
from config import BETA_RATIONALITY, EFFORT_WEIGHT


def compute_reward(action, state, goal):
    dist_before = np.linalg.norm(goal - state)
    if dist_before < 1e-3:
        return 0.0
    dist_after = np.linalg.norm(goal - (state + action))
    return (dist_before - dist_after) - EFFORT_WEIGHT * np.linalg.norm(action)


def pi_H(action, state, goal, beta=BETA_RATIONALITY):
    return np.exp(beta * compute_reward(action, state, goal))


def compute_legibility(action, state, goals, prior, target_idx):
    if np.linalg.norm(action) < 1e-3:
        return 0.0
    
    posteriors = np.array([pi_H(action, state, g) * prior[i] 
                          for i, g in enumerate(goals)])
    posteriors /= (np.sum(posteriors) + 1e-10)
    
    target_prob = posteriors[target_idx]
    other_probs = [posteriors[i] for i in range(len(goals)) if i != target_idx]
    max_other = max(other_probs) if other_probs else 1e-10
    
    if target_prob < 1e-10:
        return -10.0
    
    return np.log(target_prob + 1e-10) - np.log(max_other + 1e-10)


def optimize_legible_action(state, goals, target_idx, speed=100.0, n_samples=11):
    prior = np.ones(len(goals)) / len(goals)
    
    direction = goals[target_idx] - state
    dist = np.linalg.norm(direction)
    if dist < 1e-3:
        return np.array([0.0, 0.0])
    
    base_angle = np.arctan2(direction[1], direction[0])
    
    best_action = speed * direction / dist
    best_score = compute_legibility(best_action, state, goals, prior, target_idx)
    
    for offset in np.linspace(-1.0, 1.0, n_samples):
        angle = base_angle + offset
        candidate = speed * np.array([np.cos(angle), np.sin(angle)])
        
        legibility = compute_legibility(candidate, state, goals, prior, target_idx)
        task_score = -np.linalg.norm(goals[target_idx] - (state + candidate * 0.1))
        
        score = legibility + 0.03 * task_score
        
        if score > best_score:
            best_score = score
            best_action = candidate
    
    return best_action
