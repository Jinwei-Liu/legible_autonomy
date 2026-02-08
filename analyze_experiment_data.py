#!/usr/bin/env python3

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Nature Style Configuration
# ============================================================================
def setup_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial', 'Liberation Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
    })

# Nature-inspired color palette
COLORS = {
    'task_0.1': '#E64B35',      # Red
    'task_0.3': '#4DBBD5',      # Cyan  
    'task_0.5': '#00A087',      # Teal
    'goal_0': '#3C5488',        # Blue
    'goal_1': '#F39B7F',        # Salmon
    'trajectory': '#7E6148',    # Brown
    'user': '#B09C85',          # Light brown
    'robot': '#8491B4',         # Light blue
}

GOAL_COLORS = ['#3C5488', '#F39B7F']  # Goal 0, Goal 1
TASK_WEIGHT_COLORS = ['#E64B35', '#4DBBD5', '#00A087']  # 0.1, 0.3, 0.5

# ============================================================================
# Data Loading and Processing
# ============================================================================
def load_all_experiment_data(data_dir):
    """
    只加载每个参与者的最终数据文件，避免重复
    """
    participants = {}
    
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    
    if not json_files:
        print(f"[WARNING] No JSON files found in {data_dir}")
        return []
    
    print(f"[INFO] Found {len(json_files)} JSON files")
    
    # 按participant_id分组，只保留最完整的文件
    for filepath in sorted(json_files):
        with open(filepath, 'r') as f:
            data = json.load(f)
            data['filepath'] = filepath
            
            pid = data['participant_id']
            
            # 计算trials数量
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
            
            # 只保留最完整的文件
            if pid not in participants:
                participants[pid] = {
                    'data': data,
                    'total_trials': total_trials,
                    'file_size': file_size
                }
                print(f"  Participant {pid}: {total_trials} trials")
            else:
                stored = participants[pid]
                # 优先选择trials更多的，其次选择文件更大的
                if (total_trials > stored['total_trials'] or 
                    (total_trials == stored['total_trials'] and file_size > stored['file_size'])):
                    participants[pid]['data'] = data
                    participants[pid]['total_trials'] = total_trials
                    participants[pid]['file_size'] = file_size
                    print(f"  Participant {pid}: updated to {total_trials} trials (more complete)")
    
    final_data = [p['data'] for p in participants.values()]
    print(f"[INFO] Loaded final data for {len(final_data)} unique participant(s)")
    
    return final_data


def extract_trial_metrics(trial):
    frames = trial['frames']
    
    positions = np.array([f['position'] for f in frames])
    user_inputs = np.array([f['user_input'] for f in frames])
    robot_actions = np.array([f['robot_action'] for f in frames])
    executed_actions = np.array([f['executed'] for f in frames])
    beliefs = np.array([f['beliefs'] for f in frames])
    betas = np.array([f['beta'] for f in frames])
    
    times = [datetime.fromisoformat(f['time']) for f in frames]
    start_time = times[0]
    time_series = [(t - start_time).total_seconds() for t in times]
    
    metrics = {
        'trial_num': trial['trial_num'],
        'task_weight': trial['task_weight'],
        'target_goal_idx': trial['target_goal_idx'],
        'target_goal': trial['target_goal'],
        'duration': time_series[-1] if time_series else 0,
        'num_frames': len(frames),
        'trajectory_length': np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)),
        'avg_user_input_norm': np.mean(np.linalg.norm(user_inputs, axis=1)),
        'avg_robot_action_norm': np.mean(np.linalg.norm(robot_actions, axis=1)),
        'final_belief_0': beliefs[-1][0] if len(beliefs) > 0 else 0,
        'final_belief_1': beliefs[-1][1] if len(beliefs) > 0 else 0,
        'avg_beta': np.mean(betas),
        'positions': positions,
        'user_inputs': user_inputs,
        'robot_actions': robot_actions,
        'beliefs': beliefs,
        'betas': betas,
        'time_series': time_series,
    }
    
    return metrics


def process_participant_data(exp_data):
    participant_info = {
        'participant_id': exp_data['participant_id'],
        'gender': exp_data['gender'],
        'age': exp_data['age'],
        'experiment_date': exp_data['experiment_date'],
        'task_weight_list': exp_data.get('task_weight_list', [0.1, 0.3, 0.5]),
    }
    
    trials_metrics = []
    if 'phase1' in exp_data:
        trials = exp_data['phase1']['trials']
    elif 'trials' in exp_data:
        trials = exp_data['trials']
    else:
        print(f"[WARNING] No trials found for participant {exp_data['participant_id']}")
        trials = []
    
    for trial in trials:
        metrics = extract_trial_metrics(trial)
        trials_metrics.append(metrics)
    
    return participant_info, trials_metrics


# ============================================================================
# Individual Participant Analysis
# ============================================================================
def plot_participant_analysis(participant_info, trials_metrics, output_dir):
    setup_nature_style()
    
    participant_id = participant_info['participant_id']
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    goal_positions = {
        0: (400, 100),  # Goal 0 (Top)
        1: (400, 500),  # Goal 1 (Bottom)
    }
    
    for trial in trials_metrics:
        positions = trial['positions']
        task_weight = trial['task_weight']
        color_idx = [0.1, 0.3, 0.5].index(task_weight)
        color = TASK_WEIGHT_COLORS[color_idx]
        
        ax1.plot(positions[:, 0], positions[:, 1], 
                alpha=0.6, linewidth=1.5, color=color)
    
    for goal_idx, pos in goal_positions.items():
        ax1.scatter(pos[0], pos[1], s=200, c=GOAL_COLORS[goal_idx], 
                   marker='*', edgecolors='black', linewidths=1, 
                   label=f'Goal {goal_idx}', zorder=10)
    
    ax1.set_xlabel('X Position', fontsize=11)
    ax1.set_ylabel('Y Position', fontsize=11)
    ax1.set_title(f'(a) Trajectories - Participant {participant_id}', 
                 fontsize=12, fontweight='normal')
    ax1.legend(loc='best', frameon=False, fontsize=8)
    ax1.set_aspect('equal', adjustable='box')
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    for trial in trials_metrics[:5]:  
        time_series = trial['time_series']
        beliefs = trial['beliefs']
        
        ax2.plot(time_series, beliefs[:, 0], 
                color=GOAL_COLORS[0], alpha=0.5, linewidth=1)
        ax2.plot(time_series, beliefs[:, 1], 
                color=GOAL_COLORS[1], alpha=0.5, linewidth=1)
    
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Belief Probability', fontsize=11)
    ax2.set_title('(b) Belief Evolution', fontsize=12, fontweight='normal')
    ax2.set_ylim(-0.05, 1.05)
    
    ax3 = fig.add_subplot(gs[0, 2])
    
    for trial in trials_metrics[:5]:
        time_series = trial['time_series']
        betas = trial['betas']
        task_weight = trial['task_weight']
        color_idx = [0.1, 0.3, 0.5].index(task_weight)
        color = TASK_WEIGHT_COLORS[color_idx]
        
        ax3.plot(time_series, betas, color=color, alpha=0.6, linewidth=1)
    
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Beta (Adaptation Weight)', fontsize=11)
    ax3.set_title('(c) Adaptive Weight Evolution', fontsize=12, fontweight='normal')
    
    ax4 = fig.add_subplot(gs[1, 0])
    
    task_weights = sorted(set([t['task_weight'] for t in trials_metrics]))
    durations_by_weight = {tw: [] for tw in task_weights}
    
    for trial in trials_metrics:
        durations_by_weight[trial['task_weight']].append(trial['duration'])
    
    positions_bar = range(len(task_weights))
    means = [np.mean(durations_by_weight[tw]) for tw in task_weights]
    stds = [np.std(durations_by_weight[tw]) for tw in task_weights]
    
    bars = ax4.bar(positions_bar, means, yerr=stds, 
                   color=TASK_WEIGHT_COLORS[:len(task_weights)], 
                   alpha=0.7, capsize=5, error_kw={'linewidth': 1.5})
    
    ax4.set_xlabel('Task Weight', fontsize=11)
    ax4.set_ylabel('Duration (s)', fontsize=11)
    ax4.set_title('(d) Completion Time by Task Weight', fontsize=12, fontweight='normal')
    ax4.set_xticks(positions_bar)
    ax4.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    
    ax5 = fig.add_subplot(gs[1, 1])
    
    lengths_by_weight = {tw: [] for tw in task_weights}
    for trial in trials_metrics:
        lengths_by_weight[trial['task_weight']].append(trial['trajectory_length'])
    
    means_length = [np.mean(lengths_by_weight[tw]) for tw in task_weights]
    stds_length = [np.std(lengths_by_weight[tw]) for tw in task_weights]
    
    ax5.bar(positions_bar, means_length, yerr=stds_length,
           color=TASK_WEIGHT_COLORS[:len(task_weights)],
           alpha=0.7, capsize=5, error_kw={'linewidth': 1.5})
    
    ax5.set_xlabel('Task Weight', fontsize=11)
    ax5.set_ylabel('Trajectory Length (pixels)', fontsize=11)
    ax5.set_title('(e) Trajectory Length by Task Weight', fontsize=12, fontweight='normal')
    ax5.set_xticks(positions_bar)
    ax5.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    
    ax6 = fig.add_subplot(gs[1, 2])
    
    user_norms = [t['avg_user_input_norm'] for t in trials_metrics]
    robot_norms = [t['avg_robot_action_norm'] for t in trials_metrics]
    task_weights_all = [t['task_weight'] for t in trials_metrics]
    
    for tw, color in zip([0.1, 0.3, 0.5], TASK_WEIGHT_COLORS):
        mask = np.array(task_weights_all) == tw
        ax6.scatter(np.array(user_norms)[mask], np.array(robot_norms)[mask],
                   color=color, alpha=0.6, s=60, label=f'Weight {tw:.1f}')
    
    ax6.set_xlabel('Avg User Input Norm', fontsize=11)
    ax6.set_ylabel('Avg Robot Action Norm', fontsize=11)
    ax6.set_title('(f) User Input vs Robot Action', fontsize=12, fontweight='normal')
    ax6.legend(loc='best', frameon=False, fontsize=8)
    
    ax7 = fig.add_subplot(gs[2, 0])
    
    final_beliefs_0 = [t['final_belief_0'] for t in trials_metrics]
    final_beliefs_1 = [t['final_belief_1'] for t in trials_metrics]
    target_goals = [t['target_goal_idx'] for t in trials_metrics]
    
    for goal_idx in [0, 1]:
        mask = np.array(target_goals) == goal_idx
        beliefs = np.array(final_beliefs_0)[mask] if goal_idx == 0 else np.array(final_beliefs_1)[mask]
        ax7.hist(beliefs, bins=10, alpha=0.6, color=GOAL_COLORS[goal_idx],
                label=f'Target Goal {goal_idx}', edgecolor='black', linewidth=0.5)
    
    ax7.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax7.set_xlabel('Final Belief Probability', fontsize=11)
    ax7.set_ylabel('Frequency', fontsize=11)
    ax7.set_title('(g) Final Belief Distribution', fontsize=12, fontweight='normal')
    ax7.legend(loc='best', frameon=False, fontsize=8)
    
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    summary_stats = []
    for tw in task_weights:
        trials_tw = [t for t in trials_metrics if t['task_weight'] == tw]
        summary_stats.append([
            f'{tw:.1f}',
            f'{len(trials_tw)}',
            f'{np.mean([t["duration"] for t in trials_tw]):.2f}±{np.std([t["duration"] for t in trials_tw]):.2f}',
            f'{np.mean([t["trajectory_length"] for t in trials_tw]):.1f}±{np.std([t["trajectory_length"] for t in trials_tw]):.1f}',
            f'{np.mean([t["avg_beta"] for t in trials_tw]):.3f}±{np.std([t["avg_beta"] for t in trials_tw]):.3f}',
        ])
    
    table = ax8.table(cellText=summary_stats,
                     colLabels=['Task Weight', 'N Trials', 'Duration (s)', 'Traj. Length', 'Avg Beta'],
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.25, 0.25, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#E0E0E0')
        table[(0, i)].set_text_props(weight='bold')
    
    ax8.set_title('(h) Performance Summary Statistics', 
                 fontsize=12, fontweight='normal', pad=20)
    
    filename = f'participant_{participant_id}_analysis.pdf'
    plt.savefig(os.path.join(output_dir, filename),
               bbox_inches='tight')
    print(f"[SAVED] {filename}")
    plt.close()


# ============================================================================
# Overall Summary Analysis
# ============================================================================
def plot_overall_summary(all_participants_data, output_dir):
    setup_nature_style()
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    all_trials = []
    participant_summaries = []
    
    for exp_data in all_participants_data:
        participant_info, trials_metrics = process_participant_data(exp_data)
        all_trials.extend(trials_metrics)
        
        participant_summaries.append({
            'participant_id': participant_info['participant_id'],
            'age': participant_info['age'],
            'gender': participant_info['gender'],
            'n_trials': len(trials_metrics),
            'avg_duration': np.mean([t['duration'] for t in trials_metrics]),
            'avg_trajectory_length': np.mean([t['trajectory_length'] for t in trials_metrics]),
        })
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    task_weights = sorted(set([t['task_weight'] for t in all_trials]))
    
    data_duration = []
    for tw in task_weights:
        durations = [t['duration'] for t in all_trials if t['task_weight'] == tw]
        data_duration.append(durations)
    
    bp1 = ax1.boxplot(data_duration, positions=range(len(task_weights)), 
                      widths=0.6, patch_artist=True,
                      showfliers=True, 
                      flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5},
                      medianprops={'color': 'black', 'linewidth': 1.5})
    
    for patch, color in zip(bp1['boxes'], TASK_WEIGHT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Task Weight', fontsize=11)
    ax1.set_ylabel('Duration (s)', fontsize=11)
    ax1.set_title('(a) Completion Time Distribution', fontsize=12, fontweight='normal')
    ax1.set_xticks(range(len(task_weights)))
    ax1.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    data_length = []
    for tw in task_weights:
        lengths = [t['trajectory_length'] for t in all_trials if t['task_weight'] == tw]
        data_length.append(lengths)
    
    bp2 = ax2.boxplot(data_length, positions=range(len(task_weights)),
                      widths=0.6, patch_artist=True,
                      showfliers=True,
                      flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5},
                      medianprops={'color': 'black', 'linewidth': 1.5})
    
    for patch, color in zip(bp2['boxes'], TASK_WEIGHT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Task Weight', fontsize=11)
    ax2.set_ylabel('Trajectory Length (pixels)', fontsize=11)
    ax2.set_title('(b) Trajectory Length Distribution', fontsize=12, fontweight='normal')
    ax2.set_xticks(range(len(task_weights)))
    ax2.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    
    ax3 = fig.add_subplot(gs[0, 2])
    
    correct_beliefs = []
    for tw in task_weights:
        trials_tw = [t for t in all_trials if t['task_weight'] == tw]
        correct_count = sum(1 for t in trials_tw 
                          if (t['target_goal_idx'] == 0 and t['final_belief_0'] > 0.5) or
                             (t['target_goal_idx'] == 1 and t['final_belief_1'] > 0.5))
        accuracy = correct_count / len(trials_tw) if trials_tw else 0
        correct_beliefs.append(accuracy * 100)
    
    bars = ax3.bar(range(len(task_weights)), correct_beliefs,
                   color=TASK_WEIGHT_COLORS, alpha=0.7)
    
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=0.8, label='Chance')
    ax3.set_xlabel('Task Weight', fontsize=11)
    ax3.set_ylabel('Belief Accuracy (%)', fontsize=11)
    ax3.set_title('(c) Final Belief Accuracy', fontsize=12, fontweight='normal')
    ax3.set_xticks(range(len(task_weights)))
    ax3.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax3.set_ylim(0, 110)
    ax3.legend(loc='best', frameon=False, fontsize=8)
    
    ax4 = fig.add_subplot(gs[1, 0])
    
    participant_ids = [p['participant_id'] for p in participant_summaries]
    avg_durations = [p['avg_duration'] for p in participant_summaries]
    
    bars = ax4.bar(range(len(participant_ids)), avg_durations,
                   color='#4DBBD5', alpha=0.7)
    
    ax4.set_xlabel('Participant ID', fontsize=11)
    ax4.set_ylabel('Avg Duration (s)', fontsize=11)
    ax4.set_title('(d) Performance Across Participants', fontsize=12, fontweight='normal')
    ax4.set_xticks(range(len(participant_ids)))
    ax4.set_xticklabels(participant_ids)
    
    ax5 = fig.add_subplot(gs[1, 1])
    
    for tw, color in zip(task_weights, TASK_WEIGHT_COLORS):
        trials_tw = [t for t in all_trials if t['task_weight'] == tw]
        user_norms = [t['avg_user_input_norm'] for t in trials_tw]
        robot_norms = [t['avg_robot_action_norm'] for t in trials_tw]
        
        ax5.scatter(user_norms, robot_norms, color=color, alpha=0.5, s=50,
                   label=f'Weight {tw:.1f}')
    
    ax5.set_xlabel('Avg User Input Norm', fontsize=11)
    ax5.set_ylabel('Avg Robot Action Norm', fontsize=11)
    ax5.set_title('(e) User-Robot Action Relationship', fontsize=12, fontweight='normal')
    ax5.legend(loc='best', frameon=False, fontsize=8)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    overall_stats = []
    for tw in task_weights:
        trials_tw = [t for t in all_trials if t['task_weight'] == tw]
        overall_stats.append([
            f'{tw:.1f}',
            f'{len(trials_tw)}',
            f'{np.mean([t["duration"] for t in trials_tw]):.2f}±{np.std([t["duration"] for t in trials_tw]):.2f}',
            f'{np.mean([t["trajectory_length"] for t in trials_tw]):.1f}±{np.std([t["trajectory_length"] for t in trials_tw]):.1f}',
        ])
    
    table = ax6.table(cellText=overall_stats,
                     colLabels=['Task Weight', 'N Trials', 'Duration (s)', 'Traj. Length'],
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#E0E0E0')
        table[(0, i)].set_text_props(weight='bold')
    
    ax6.set_title('(f) Overall Statistics Summary', 
                 fontsize=12, fontweight='normal', pad=20)
    
    plt.savefig(os.path.join(output_dir, 'overall_summary.pdf'), bbox_inches='tight')
    print(f"[SAVED] overall_summary.pdf")
    plt.close()


def print_overall_statistics(all_participants_data):
    print("\n" + "="*70)
    print("  OVERALL EXPERIMENT STATISTICS")
    print("="*70)
    
    all_trials = []
    for exp_data in all_participants_data:
        _, trials_metrics = process_participant_data(exp_data)
        all_trials.extend(trials_metrics)
    
    print(f"\n[1] Overall information")
    print(f"  Participants: {len(all_participants_data)}")
    print(f"  Total trials: {len(all_trials)}")
    
    task_weights = sorted(set([t['task_weight'] for t in all_trials]))
    print(f"\n[2] Statistics by task weight")
    for tw in task_weights:
        trials_tw = [t for t in all_trials if t['task_weight'] == tw]
        print(f"\n  Task Weight {tw:.1f}:")
        print(f"    Trials: {len(trials_tw)}")
        print(f"    Duration: {np.mean([t['duration'] for t in trials_tw]):.2f}±{np.std([t['duration'] for t in trials_tw]):.2f}s")
        print(f"    Trajectory length: {np.mean([t['trajectory_length'] for t in trials_tw]):.1f}±{np.std([t['trajectory_length'] for t in trials_tw]):.1f}")
        
        correct = sum(1 for t in trials_tw 
                     if (t['target_goal_idx'] == 0 and t['final_belief_0'] > 0.5) or
                        (t['target_goal_idx'] == 1 and t['final_belief_1'] > 0.5))
        accuracy = correct / len(trials_tw) * 100 if trials_tw else 0
        print(f"    Belief accuracy: {accuracy:.1f}%")
    
    print("\n" + "="*70)


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze legible autonomy experiment data')
    parser.add_argument('--input', '-i', type=str, 
                       default='./experiment_data',
                       help='Input directory containing JSON files')
    parser.add_argument('--output', '-o', type=str, 
                       default='./experiment_results',
                       help='Output directory for figures')
    parser.add_argument('--stats-only', action='store_true', 
                       help='Print statistics only')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    all_data = load_all_experiment_data(args.input)
    
    if len(all_data) == 0:
        print("[ERROR] No data files found!")
        return
    
    print_overall_statistics(all_data)
    
    if args.stats_only:
        return
    
    print("\n[Generating figures...]")
    
    for idx, exp_data in enumerate(all_data, start=1):
        participant_info, trials_metrics = process_participant_data(exp_data)
        plot_participant_analysis(participant_info, trials_metrics, args.output)
    
    plot_overall_summary(all_data, args.output)
    
    print(f"\n[DONE] All figures saved to: {args.output}/")


if __name__ == '__main__':
    main()