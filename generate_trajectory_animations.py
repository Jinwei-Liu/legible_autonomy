#!/usr/bin/env python3
"""
Generate animated GIF visualizations for each participant's trajectories
Shows trajectory evolution over time with participant number
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Correct goal positions from experiment_collection.py
GOAL_POSITIONS = {
    0: (650.0, 290.0),  # Goal 0 (Right-Top)
    1: (680.0, 310.0),  # Goal 1 (Right-Bottom)
}

START_POSITION = (100.0, 300.0)
CANVAS_SIZE = (800, 600)  # WIDTH, HEIGHT

# Color scheme
GOAL_COLORS = ['#3C5488', '#F39B7F']  # Goal 0 (Blue), Goal 1 (Salmon)
TASK_WEIGHT_COLORS = {
    0: '#E64B35',   # Red
    5: '#4DBBD5',   # Cyan  
    10: '#00A087',  # Teal
}
TRAJECTORY_COLOR = '#7E6148'  # Brown
START_COLOR = '#00A087'  # Teal

# Animation parameters
FPS = 10  # Frames per second
TRAIL_LENGTH = 15  # Number of recent positions to show in trail
SPEED_MULTIPLIER = 5  # Animation speed multiplier (e.g., 5 means 5x speed)


# ============================================================================
# Data Loading
# ============================================================================

def load_participant_data(data_dir):
    """
    Load all participant experiment data
    Returns: dict with participant_id as key
    """
    participants = {}
    
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    
    if not json_files:
        print(f"[WARNING] No JSON files found in {data_dir}")
        return {}
    
    print(f"[INFO] Found {len(json_files)} JSON files")
    
    # Group by participant_id and keep most complete file
    for filepath in sorted(json_files):
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            pid = data['participant_id']
            
            # Count trials
            phase1_trials = 0
            if 'phase1' in data and 'trials' in data['phase1']:
                phase1_trials = len(data['phase1']['trials'])
            elif 'trials' in data:
                phase1_trials = len(data['trials'])
            
            phase2_trials = 0
            if 'phase2' in data and 'trials' in data['phase2']:
                phase2_trials = len(data['phase2']['trials'])
            
            total_trials = phase1_trials + phase2_trials
            file_size = os.path.getsize(filepath)
            
            # Keep most complete file
            if pid not in participants:
                participants[pid] = {
                    'data': data,
                    'total_trials': total_trials,
                    'file_size': file_size
                }
                print(f"  Participant {pid}: {total_trials} trials")
            else:
                stored = participants[pid]
                if (total_trials > stored['total_trials'] or 
                    (total_trials == stored['total_trials'] and file_size > stored['file_size'])):
                    participants[pid]['data'] = data
                    participants[pid]['total_trials'] = total_trials
                    participants[pid]['file_size'] = file_size
    
    return {pid: p['data'] for pid, p in participants.items()}


def extract_trials(exp_data):
    """Extract trials from experiment data"""
    if 'phase1' in exp_data:
        trials = exp_data['phase1']['trials']
    elif 'trials' in exp_data:
        trials = exp_data['trials']
    else:
        trials = []
    
    return trials


# ============================================================================
# Animation Generation
# ============================================================================

def create_trajectory_animation(participant_id, trials, output_dir, dpi=100, speed_multiplier=5):
    """
    Create animated GIF showing all trajectories for one participant
    All trajectories animate simultaneously
    
    Args:
        participant_id: Participant identifier
        trials: List of trial data
        output_dir: Directory to save output GIF
        dpi: Resolution of the output
        speed_multiplier: Animation speed multiplier (e.g., 5 means 5x speed)
    """
    
    print(f"\n[Processing] Participant {participant_id}")
    print(f"  Total trials: {len(trials)}")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor='white')
    
    # Collect all trajectories with timestamps
    all_trajectories = []
    for trial in trials:
        frames = trial['frames']
        positions = np.array([f['position'] for f in frames])
        
        # Extract timestamps and convert to seconds from start
        from datetime import datetime
        times = [datetime.fromisoformat(f['time']) for f in frames]
        start_time = times[0]
        time_series = np.array([(t - start_time).total_seconds() for t in times])
        
        task_weight = trial['task_weight']
        target_goal_idx = trial['target_goal_idx']
        
        all_trajectories.append({
            'positions': positions,
            'time_series': time_series,
            'task_weight': task_weight,
            'target_goal_idx': target_goal_idx,
            'trial_num': trial['trial_num'],
            'duration': time_series[-1] if len(time_series) > 0 else 0
        })
    
    print(f"  Extracted {len(all_trajectories)} trajectories")
    
    # Calculate total animation time based on longest trajectory duration
    max_duration = max(traj['duration'] for traj in all_trajectories)
    
    # Calculate frames based on real time
    # Animation time = max_duration / speed_multiplier
    animation_duration = max_duration / speed_multiplier
    total_frames = int(animation_duration * FPS) + 20  # Extra frames at end
    
    print(f"  Max trajectory duration: {max_duration:.2f}s")
    print(f"  Animation duration at {speed_multiplier}x speed: {animation_duration:.2f}s")
    print(f"  Generating {total_frames} frames at {FPS} fps...")
    
    def init():
        """Initialize animation"""
        ax.clear()
        
        # Set up canvas
        ax.set_xlim(0, CANVAS_SIZE[0])
        ax.set_ylim(0, CANVAS_SIZE[1])
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match pygame coordinate system
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Draw start position
        start_circle = Circle(START_POSITION, 15, color=START_COLOR, alpha=0.3, 
                            label='Start', zorder=1)
        ax.add_patch(start_circle)
        ax.plot(START_POSITION[0], START_POSITION[1], 'o', 
               color=START_COLOR, markersize=8, zorder=2)
        
        # Draw goals
        for goal_idx, goal_pos in GOAL_POSITIONS.items():
            goal_circle = Circle(goal_pos, 19, color=GOAL_COLORS[goal_idx], 
                               alpha=0.3, zorder=1)
            ax.add_patch(goal_circle)
            ax.plot(goal_pos[0], goal_pos[1], 'o', 
                   color=GOAL_COLORS[goal_idx], markersize=10, zorder=2)
            ax.text(goal_pos[0] + 30, goal_pos[1], f'Goal {goal_idx}',
                   fontsize=11, color=GOAL_COLORS[goal_idx], 
                   fontweight='bold', va='center')
        
        # Title with participant number only
        ax.set_title(f'Participant {participant_id}',
                    fontsize=14, fontweight='bold', pad=15)
        
        return []
    
    def animate(frame_num):
        """Animation function - all trajectories move simultaneously based on real time"""
        
        # Clear previous trajectory elements (but keep goals and start)
        for artist in ax.lines + ax.collections:
            if artist not in ax.patches:
                artist.remove()
        
        # Remove old text (except title and goal labels)
        for text in ax.texts[3:]:  # Keep first 3 texts (goal labels)
            text.remove()
        
        # Calculate current real time (accounting for speed multiplier)
        current_time = (frame_num / FPS) * speed_multiplier
        
        # Draw all trajectories up to current time
        for traj in all_trajectories:
            positions = traj['positions']
            time_series = traj['time_series']
            task_weight = traj['task_weight']
            color = TASK_WEIGHT_COLORS.get(task_weight, TRAJECTORY_COLOR)
            
            # Find which frames to display based on current_time
            valid_indices = np.where(time_series <= current_time)[0]
            
            if len(valid_indices) > 0:
                # Draw completed path up to current time
                current_idx = valid_indices[-1]
                
                ax.plot(positions[:current_idx+1, 0], positions[:current_idx+1, 1], '-',
                       color=color, linewidth=2, alpha=0.7, zorder=3)
                
                # Draw trail (recent positions)
                trail_start = max(0, current_idx - TRAIL_LENGTH)
                trail_positions = positions[trail_start:current_idx+1]
                
                # Fade trail
                for i in range(len(trail_positions) - 1):
                    alpha = 0.3 + 0.7 * (i / max(1, len(trail_positions) - 1))
                    ax.plot(trail_positions[i:i+2, 0], trail_positions[i:i+2, 1], '-',
                           color=color, linewidth=3, alpha=alpha, zorder=4)
                
                # Draw current position (robot) only if trajectory is still active
                if current_time < time_series[-1]:
                    # Interpolate position if between frames
                    if current_idx < len(positions) - 1:
                        # Linear interpolation between current and next frame
                        t1, t2 = time_series[current_idx], time_series[current_idx + 1]
                        p1, p2 = positions[current_idx], positions[current_idx + 1]
                        
                        # Interpolation factor
                        alpha_interp = (current_time - t1) / (t2 - t1) if t2 > t1 else 0
                        alpha_interp = np.clip(alpha_interp, 0, 1)
                        
                        current_pos = p1 + alpha_interp * (p2 - p1)
                    else:
                        current_pos = positions[current_idx]
                    
                    ax.plot(current_pos[0], current_pos[1], 'o',
                           color=color, markersize=12, markeredgecolor='black', 
                           markeredgewidth=2, zorder=5)
        
        # Speed multiplier indicator (top-right corner)
        ax.text(CANVAS_SIZE[0] - 10, 30, f'×{speed_multiplier}',
               fontsize=12, ha='right', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        # Current time indicator (bottom-left corner)
        ax.text(10, CANVAS_SIZE[1] - 10, f'Time: {current_time:.2f}s',
               fontsize=9, va='bottom', color='gray')
        
        return []
    
    # Create animation
    init()
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=total_frames, interval=1000/FPS,
                                  blit=False, repeat=True)
    
    # Save as GIF
    output_path = os.path.join(output_dir, f'participant_{participant_id}_trajectories.gif')
    
    print(f"  Saving animation to {output_path}...")
    writer = animation.PillowWriter(fps=FPS)
    anim.save(output_path, writer=writer, dpi=dpi)
    
    plt.close(fig)
    
    print(f"  [SAVED] {output_path}")
    
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate animated GIF visualizations for participant trajectories'
    )
    parser.add_argument('--input', '-i', type=str, 
                       default='./experiment_data',
                       help='Input directory containing JSON files')
    parser.add_argument('--output', '-o', type=str, 
                       default='./trajectory_animations',
                       help='Output directory for GIF animations')
    parser.add_argument('--dpi', type=int, default=100,
                       help='DPI for output GIFs (default: 100)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second (default: 10)')
    parser.add_argument('--speed', type=float, default=5,
                       help='Animation speed multiplier (default: 5)')
    
    args = parser.parse_args()
    
    # Update global FPS if specified
    global FPS, SPEED_MULTIPLIER
    FPS = args.fps
    SPEED_MULTIPLIER = args.speed
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load all participant data
    print("\n" + "="*70)
    print("  TRAJECTORY ANIMATION GENERATOR")
    print("="*70)
    
    participants = load_participant_data(args.input)
    
    if not participants:
        print("[ERROR] No participant data found!")
        return
    
    print(f"\n[INFO] Loaded {len(participants)} participant(s)")
    
    # Generate animation for each participant
    for participant_id, exp_data in participants.items():
        trials = extract_trials(exp_data)
        
        if not trials:
            print(f"[WARNING] No trials found for participant {participant_id}, skipping")
            continue
        
        create_trajectory_animation(participant_id, trials, args.output, 
                                          dpi=args.dpi, speed_multiplier=args.speed)
    
    print("\n" + "="*70)
    print(f"[DONE] All animations saved to: {args.output}/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()