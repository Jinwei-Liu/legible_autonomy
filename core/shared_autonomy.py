import numpy as np
from core.legibility import pi_H, compute_legibility, compute_reward
from config import (ROBOT_SPEED, BETA_BASE, BETA_MIN, B_THRESH, 
                   D_MIN, D_MAX, TASK_WEIGHT, SEARCH_ANGLES, ANGLE_RANGE, 
                   WIDTH, HEIGHT, BETA_RATIONALITY, USER_SPEED)


class LegibleSharedAutonomy:
    def __init__(self, goals, task_weight=None):
        self.goals = goals
        self.beliefs = np.ones(len(goals)) / len(goals)
        self.beta = BETA_BASE
        self.d_workspace = np.sqrt(WIDTH**2 + HEIGHT**2)
        # Allow overriding task_weight for experiments
        self.task_weight = task_weight if task_weight is not None else TASK_WEIGHT
    
    def update_belief(self, state, user_input):
        """Update belief using absolute rewards (CORRECTED)"""
        if np.linalg.norm(user_input) < 0.01:
            return
        
        # Compute likelihood for each goal using ABSOLUTE reward
        # P(θ|a) ∝ exp(β·R(a,s,θ)) · P(θ)
        log_likelihoods = []
        for g in self.goals:
            user_reward = compute_reward(user_input, state, g)
            log_likelihoods.append(BETA_RATIONALITY * user_reward)
        
        log_likelihoods = np.array(log_likelihoods)
        
        # Update beliefs in log space for numerical stability
        log_beliefs = np.log(self.beliefs + 1e-100)
        log_beliefs += log_likelihoods
        
        # Normalize
        max_log = np.max(log_beliefs)
        log_beliefs -= max_log
        
        self.beliefs = np.exp(log_beliefs)
        self.beliefs /= np.sum(self.beliefs)
    
    def compute_adaptive_beta(self, state, target_goal):
        b_max = np.max(self.beliefs)
        dist = np.linalg.norm(target_goal - state)
        
        alpha = np.clip((b_max - B_THRESH) / (1.0 - B_THRESH), 0.0, 1.0)
        
        d_min_abs, d_max_abs = D_MIN * self.d_workspace, D_MAX * self.d_workspace
        gamma = np.clip((d_max_abs - dist) / (d_max_abs - d_min_abs), 0.0, 1.0)
        
        return BETA_BASE - alpha * gamma * (BETA_BASE - BETA_MIN)
    
    def compute_robot_action(self, state, user_input):
        """Compute legible robot action using simplified optimization"""
        target_idx = np.argmax(self.beliefs)
        target_goal = self.goals[target_idx]
        
        direction = target_goal - state
        dist = np.linalg.norm(direction)
        if dist < 1e-3:
            return np.array([0.0, 0.0])
        
        task_action = ROBOT_SPEED * direction / dist
        self.beta = self.compute_adaptive_beta(state, target_goal)
        
        best_action, best_score = task_action, -np.inf
        base_angle = np.arctan2(task_action[1], task_action[0])
        
        # Search over candidate actions
        for offset in np.linspace(-ANGLE_RANGE, ANGLE_RANGE, SEARCH_ANGLES):
            angle = base_angle + offset
            candidate = ROBOT_SPEED * np.array([np.cos(angle), np.sin(angle)])
            
            # Compute legibility score
            leg_score = compute_legibility(candidate, state, self.goals, 
                                          self.beliefs, target_idx)
            
            # Compute task performance (negative distance)
            task_score = -np.linalg.norm(target_goal - (state + candidate * 0.1))
            
            # CORRECTED: Use simplified form L + λ·Q
            score = leg_score + self.task_weight * task_score
            
            if score > best_score:
                best_score, best_action = score, candidate
        
        return best_action