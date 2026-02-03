"""
Experiment Data Collection for Legible Shared Autonomy
In-game questionnaire with D-pad control
"""

import pygame
import numpy as np
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from datetime import datetime
import random

sys.path.append('..')

from core.shared_autonomy import LegibleSharedAutonomy
from utils.visualization import draw_arrow, draw_goal
from config import (WIDTH, HEIGHT, BLACK, WHITE, RED, GREEN, BLUE, YELLOW, CYAN, 
                   USER_SPEED, TASK_WEIGHT_LIST, TRIALS_PER_CONDITION, MIDPOINT_THRESHOLD)


class ParticipantDialog:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Participant Information")
        self.root.geometry("400x250")
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        self.root.focus_force()
        
        self.participant_id = None
        self.gender = None
        self.age = None
        
        tk.Label(self.root, text="Participant Code:", font=("Arial", 12)).pack(pady=5)
        self.id_entry = tk.Entry(self.root, font=("Arial", 12), width=20)
        self.id_entry.pack(pady=5)
        
        tk.Label(self.root, text="Gender:", font=("Arial", 12)).pack(pady=5)
        self.gender_var = tk.StringVar()
        gender_combo = ttk.Combobox(self.root, textvariable=self.gender_var, 
                                    values=["Male", "Female", "Other", "Prefer not to say"],
                                    font=("Arial", 12), width=18, state="readonly")
        gender_combo.pack(pady=5)
        gender_combo.current(0)
        
        tk.Label(self.root, text="Age:", font=("Arial", 12)).pack(pady=5)
        self.age_entry = tk.Entry(self.root, font=("Arial", 12), width=20)
        self.age_entry.pack(pady=5)
        
        tk.Button(self.root, text="Start Experiment", command=self.submit, 
                 font=("Arial", 12), bg="green", fg="white", width=20).pack(pady=15)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def submit(self):
        self.participant_id = self.id_entry.get().strip()
        self.gender = self.gender_var.get()
        age_str = self.age_entry.get().strip()
        
        if not self.participant_id:
            messagebox.showerror("Error", "Please enter participant code")
            return
        
        try:
            self.age = int(age_str)
            if self.age < 0 or self.age > 120:
                raise ValueError()
        except:
            messagebox.showerror("Error", "Please enter a valid age (0-120)")
            return
        
        self.root.quit()
        self.root.destroy()
    
    def on_close(self):
        self.root.quit()
        self.root.destroy()
        sys.exit()
    
    def get_info(self):
        self.root.mainloop()
        return self.participant_id, self.gender, self.age


class InGameQuestionnaire:
    def __init__(self, screen, goal_names, x_offset=10):
        self.screen = screen
        self.goal_names = goal_names
        self.x_offset = x_offset
        
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        
        self.question_index = 0
        self.understood_selection = 0
        self.goal_selection = 0
        
        self.understood = None
        self.predicted_goal = None
        self.completed = False
    
    def handle_input(self, event):
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == 11:  # D-pad Up
                self.move_selection(-1)
            elif event.button == 12:  # D-pad Down
                self.move_selection(1)
            elif event.button == 1:  # Circle
                self.confirm_selection()
    
    def move_selection(self, direction):
        if self.question_index == 0:
            self.understood_selection = (self.understood_selection + direction) % 2
        else:
            self.goal_selection = (self.goal_selection + direction) % len(self.goal_names)
    
    def confirm_selection(self):
        if self.question_index == 0:
            self.understood = "Yes" if self.understood_selection == 0 else "No"
            self.question_index = 1
        else:
            self.predicted_goal = self.goal_names[self.goal_selection]
            self.completed = True
    
    def draw(self):
        panel_rect = pygame.Rect(self.x_offset - 10, 50, 270, HEIGHT - 100)
        pygame.draw.rect(self.screen, (30, 30, 30), panel_rect)
        pygame.draw.rect(self.screen, CYAN, panel_rect, 2)
        
        y = 70
        
        title_text = "Questionnaire"
        title_surface = self.font_large.render(title_text, True, YELLOW)
        self.screen.blit(title_surface, (self.x_offset, y))
        y += 50
        
        if self.question_index == 0:
            q1_text = "Did robot understand"
            q1_text2 = "your intention?"
            
            text1 = self.font_medium.render(q1_text, True, WHITE)
            text2 = self.font_medium.render(q1_text2, True, WHITE)
            
            self.screen.blit(text1, (self.x_offset, y))
            y += 30
            self.screen.blit(text2, (self.x_offset, y))
            y += 50
            
            options = ["Yes", "No"]
            for i, option in enumerate(options):
                color = CYAN if i == self.understood_selection else WHITE
                marker = ">" if i == self.understood_selection else " "
                
                option_text = f"{marker} {option}"
                text_surface = self.font_medium.render(option_text, True, color)
                self.screen.blit(text_surface, (self.x_offset + 20, y))
                y += 35
            
        else:
            result_text = f"Understood: {self.understood}"
            result_surface = self.font_small.render(result_text, True, (150, 150, 150))
            self.screen.blit(result_surface, (self.x_offset, y))
            y += 40
            
            q2_text = "Robot's predicted"
            q2_text2 = "goal?"
            
            text1 = self.font_medium.render(q2_text, True, WHITE)
            text2 = self.font_medium.render(q2_text2, True, WHITE)
            
            self.screen.blit(text1, (self.x_offset, y))
            y += 30
            self.screen.blit(text2, (self.x_offset, y))
            y += 50
            
            for i, goal_name in enumerate(self.goal_names):
                color = CYAN if i == self.goal_selection else WHITE
                marker = ">" if i == self.goal_selection else " "
                
                display_name = goal_name.replace("Goal ", "G").replace(" (Top)", "").replace(" (Bottom)", "")
                
                option_text = f"{marker} {display_name}"
                text_surface = self.font_medium.render(option_text, True, color)
                self.screen.blit(text_surface, (self.x_offset + 20, y))
                y += 35
        
        y = HEIGHT - 80
        instr_text = "D-pad: Up/Down"
        instr_surface = self.font_small.render(instr_text, True, (200, 200, 200))
        self.screen.blit(instr_surface, (self.x_offset, y))
        
        y += 25
        confirm_text = "Press ○ to confirm"
        confirm_surface = self.font_small.render(confirm_text, True, (200, 200, 200))
        self.screen.blit(confirm_surface, (self.x_offset, y))


class ExperimentSession:
    def __init__(self, participant_id, gender, age):
        self.participant_id = participant_id
        self.gender = gender
        self.age = age
        
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Legible Shared Autonomy Experiment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Controller connected: {self.joystick.get_name()}")
        else:
            messagebox.showwarning("Warning", "No PS5 controller detected!")
        
        self.goals = [
            np.array([650.0, 290.0]),
            np.array([680.0, 310.0])
        ]
        self.goal_names = ["Goal 0 (Top)", "Goal 1 (Bottom)"]
        self.start_pos = np.array([100.0, 300.0])
        
        self.trial_sequence = self.generate_trial_sequence()
        self.current_trial = 0
        self.total_trials = len(self.trial_sequence)
        
        self.all_trial_data = []
        
        self.data_dir = "experiment_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def generate_trial_sequence(self):
        trials = []
        for task_weight in TASK_WEIGHT_LIST:
            for trial_num in range(TRIALS_PER_CONDITION):
                target_goal_idx = random.choice([0, 1])
                trials.append({
                    'task_weight': task_weight,
                    'trial_num': trial_num,
                    'target_goal_idx': target_goal_idx
                })
        random.shuffle(trials)
        return trials
    
    def run_trial(self, trial_info):
        task_weight = trial_info['task_weight']
        target_goal_idx = trial_info['target_goal_idx']
        target_goal = self.goals[target_goal_idx]
        
        robot_pos = self.start_pos.copy()
        sa = LegibleSharedAutonomy(self.goals)
        sa.task_weight = task_weight
        
        self.show_instruction_screen(target_goal_idx, task_weight)
        
        trial_data = {
            'trial_num': self.current_trial + 1,
            'task_weight': task_weight,
            'target_goal_idx': target_goal_idx,
            'target_goal': self.goal_names[target_goal_idx],
            'start_time': datetime.now().isoformat(),
            'frames': []
        }
        
        running = True
        paused = False
        midpoint_triggered = False
        questionnaire = None
        
        initial_distance = np.linalg.norm(target_goal - robot_pos)
        
        while running:
            dt = self.clock.tick(30) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                if questionnaire and not questionnaire.completed:
                    questionnaire.handle_input(event)
                    
                    if questionnaire.completed:
                        current_distance = np.linalg.norm(target_goal - robot_pos)
                        progress = 1 - (current_distance / initial_distance)
                        
                        questionnaire_data = {
                            'understood': questionnaire.understood,
                            'predicted_goal': questionnaire.predicted_goal,
                            'actual_goal': self.goal_names[target_goal_idx],
                            'progress': progress
                        }
                        trial_data['questionnaire'] = questionnaire_data
                        
                        questionnaire = None
                        paused = False
            
            if not paused:
                user_input = np.array([0.0, 0.0])
                
                if self.joystick:
                    x_axis = self.joystick.get_axis(0)
                    y_axis = self.joystick.get_axis(1)
                    
                    if abs(x_axis) > 0.1 or abs(y_axis) > 0.1:
                        user_input = np.array([x_axis, y_axis]) * USER_SPEED
                
                sa.update_belief(robot_pos, user_input)
                robot_action = sa.compute_robot_action(robot_pos, user_input)
                
                if np.linalg.norm(user_input) < 0.01:
                    executed = robot_action
                else:
                    executed = sa.beta * user_input + (1 - sa.beta) * robot_action
                
                robot_pos += executed * dt
                robot_pos = np.clip(robot_pos, [0, 0], [WIDTH, HEIGHT])
                
                trial_data['frames'].append({
                    'time': datetime.now().isoformat(),
                    'position': robot_pos.tolist(),
                    'user_input': user_input.tolist(),
                    'robot_action': robot_action.tolist(),
                    'executed': executed.tolist(),
                    'beliefs': sa.beliefs.tolist(),
                    'beta': sa.beta
                })
                
                current_distance = np.linalg.norm(target_goal - robot_pos)
                progress = 1 - (current_distance / initial_distance)
                
                if not midpoint_triggered and progress >= MIDPOINT_THRESHOLD:
                    midpoint_triggered = True
                    paused = True
                    questionnaire = InGameQuestionnaire(self.screen, self.goal_names)
            
            if np.linalg.norm(target_goal - robot_pos) < 30:
                running = False
            
            self.screen.fill(BLACK)
            
            for i, goal in enumerate(self.goals):
                draw_goal(self.screen, goal, 15, GREEN, f"G{i}", self.font_small)
            
            pygame.draw.circle(self.screen, RED, robot_pos.astype(int), 10)
            
            if not paused:
                scale = 0.5
                if np.linalg.norm(user_input) > 1:
                    draw_arrow(self.screen, robot_pos, robot_pos + user_input * scale, WHITE, 2)
                if 'executed' in locals() and np.linalg.norm(executed) > 1:
                    draw_arrow(self.screen, robot_pos, robot_pos + executed * scale, YELLOW, 2)
            
            target_text = f"YOUR GOAL: {self.goal_names[target_goal_idx]}"
            text_surface = self.font.render(target_text, True, CYAN)
            bg_rect = text_surface.get_rect(topleft=(10, 10))
            bg_rect.inflate_ip(20, 10)
            pygame.draw.rect(self.screen, (50, 50, 50), bg_rect)
            self.screen.blit(text_surface, (15, 15))
            
            info_text = f"Trial {self.current_trial + 1}/{self.total_trials}"
            text_surface = self.font_small.render(info_text, True, WHITE)
            self.screen.blit(text_surface, (10, HEIGHT - 30))
            
            if questionnaire:
                questionnaire.draw()
            
            pygame.display.flip()
        
        trial_data['end_time'] = datetime.now().isoformat()
        trial_data['final_position'] = robot_pos.tolist()
        
        return trial_data
    
    def show_instruction_screen(self, target_goal_idx, task_weight):
        self.screen.fill(BLACK)
        
        lines = [
            f"Trial {self.current_trial + 1} / {self.total_trials}",
            "",
            f"YOUR GOAL: {self.goal_names[target_goal_idx]}",
            "",
            "Use LEFT STICK to control",
            "",
            "Press ○ to start"
        ]
        
        y = 150
        for line in lines:
            if "YOUR GOAL" in line:
                color = CYAN
                font = self.font
            else:
                color = WHITE
                font = self.font_small
            
            text_surface = font.render(line, True, color)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, y))
            self.screen.blit(text_surface, text_rect)
            y += 50 if font == self.font else 35
        
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 1:
                        waiting = False
    
    def run_experiment(self):
        for trial_info in self.trial_sequence:
            self.current_trial = self.trial_sequence.index(trial_info)
            
            trial_data = self.run_trial(trial_info)
            
            if trial_data is None:
                break
            
            self.all_trial_data.append(trial_data)
            self.save_data()
        
        self.show_completion_screen()
        
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
    
    def save_data(self):
        data = {
            'participant_id': self.participant_id,
            'gender': self.gender,
            'age': self.age,
            'experiment_date': datetime.now().isoformat(),
            'task_weight_list': TASK_WEIGHT_LIST,
            'trials_per_condition': TRIALS_PER_CONDITION,
            'trials': self.all_trial_data
        }
        
        filename = f"{self.data_dir}/participant_{self.participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data saved to {filename}")
    
    def show_completion_screen(self):
        self.screen.fill(BLACK)
        
        lines = [
            "Experiment Complete!",
            "",
            "Thank you for your participation.",
            "",
            f"Total trials completed: {len(self.all_trial_data)}",
            "",
            "Press any button to exit"
        ]
        
        y = 150
        for i, line in enumerate(lines):
            if i == 0:
                color = GREEN
                font = self.font
            else:
                color = WHITE
                font = self.font_small
            
            text_surface = font.render(line, True, color)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, y))
            self.screen.blit(text_surface, text_rect)
            y += 50 if font == self.font else 35
        
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.JOYBUTTONDOWN:
                    waiting = False


def main():
    dialog = ParticipantDialog()
    participant_id, gender, age = dialog.get_info()
    
    if participant_id is None:
        print("Experiment cancelled")
        return
    
    print(f"Starting experiment for participant: {participant_id}")
    print(f"Gender: {gender}, Age: {age}")
    print(f"Testing {len(TASK_WEIGHT_LIST)} conditions × {TRIALS_PER_CONDITION} trials = {len(TASK_WEIGHT_LIST) * TRIALS_PER_CONDITION} total trials")
    
    session = ExperimentSession(participant_id, gender, age)
    session.run_experiment()
    
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()
