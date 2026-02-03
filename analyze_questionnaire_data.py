#!/usr/bin/env python3
"""
Questionnaire Data Analysis Script (Task Weight Focused)
Analyzes questionnaire responses grouped by task weight
Style: Nature journal figure aesthetics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path

# Nature journal style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3

# Color palette (Nature-inspired)
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent1': '#F18F01',
    'accent2': '#C73E1D',
    'neutral': '#6C757D',
    'success': '#06A77D',
    'warning': '#F77F00',
}

def load_experiment_data(data_dir='experiment_data'):
    """Load all experiment data from JSON files"""
    data_path = Path(data_dir)
    all_data = []
    
    # Group files by participant ID
    participant_files = {}
    for json_file in data_path.glob('participant_*.json'):
        # Extract participant ID from filename
        parts = json_file.stem.split('_')
        if len(parts) >= 2:
            pid = parts[1]
            if pid not in participant_files:
                participant_files[pid] = []
            participant_files[pid].append(json_file)
    
    # For each participant, only load the LATEST (largest) file
    for pid, files in participant_files.items():
        # Sort by file size (larger = more complete) or by timestamp in filename
        latest_file = max(files, key=lambda f: f.stat().st_size)
        with open(latest_file, 'r') as f:
            participant_data = json.load(f)
            all_data.append(participant_data)
            print(f"   Loaded {latest_file.name} (most complete file for participant {pid})")
    
    return all_data

def extract_questionnaire_data(all_data):
    """Extract questionnaire responses from experiment data"""
    questionnaire_data = []
    
    for participant in all_data:
        participant_id = participant['participant_id']
        
        for trial in participant['trials']:
            if 'questionnaire' in trial:
                q_data = trial['questionnaire']
                questionnaire_data.append({
                    'participant_id': participant_id,
                    'trial_number': trial['trial_num'],
                    'task_weight': trial['task_weight'],
                    'actual_goal': trial['target_goal'],
                    'understood': q_data.get('understood', 'unknown'),
                    'predicted_goal': q_data.get('predicted_goal', None),
                    'timestamp': q_data.get('timestamp', 0)
                })
    
    return questionnaire_data

def analyze_participant_questionnaire(participant_id, questionnaire_data, output_dir='questionnaire_results'):
    """Create comprehensive questionnaire analysis for a single participant (by task weight)"""
    
    # Filter data for this participant
    p_data = [q for q in questionnaire_data if q['participant_id'] == participant_id]
    
    if not p_data:
        print(f"No questionnaire data found for participant {participant_id}")
        return
    
    # Get unique task weights
    task_weights = sorted(set(q['task_weight'] for q in p_data))
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.95, top=0.93, bottom=0.05)
    
    # (a) Understanding (Yes/No) by Task Weight
    ax1 = fig.add_subplot(gs[0, 0])
    yes_counts = []
    no_counts = []
    
    for tw in task_weights:
        tw_data = [q for q in p_data if q['task_weight'] == tw]
        yes_count = sum(1 for q in tw_data if q['understood'].lower() == 'yes')
        no_count = sum(1 for q in tw_data if q['understood'].lower() == 'no')
        yes_counts.append(yes_count)
        no_counts.append(no_count)
    
    x = np.arange(len(task_weights))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, yes_counts, width, label='Yes', 
                    color=COLORS['success'], alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, no_counts, width, label='No',
                    color=COLORS['warning'], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Task Weight', fontsize=9)
    ax1.set_ylabel('Count', fontsize=9)
    ax1.set_title('(a) Understanding of Robot Intent by Task Weight', fontsize=10, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax1.legend(fontsize=8, frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=7)
    
    # (b) Prediction accuracy by Task Weight
    ax2 = fig.add_subplot(gs[0, 1])
    accuracy_by_weight = []
    count_by_weight = []
    
    for tw in task_weights:
        tw_data = [q for q in p_data if q['task_weight'] == tw]
        correct = sum(1 for q in tw_data if q['predicted_goal'] == q['actual_goal'])
        total = len([q for q in tw_data if q['predicted_goal'] is not None])
        acc = (correct / total * 100) if total > 0 else 0
        accuracy_by_weight.append(acc)
        count_by_weight.append(total)
    
    bars = ax2.bar(range(len(task_weights)), accuracy_by_weight,
                   color=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(task_weights)))
    ax2.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax2.set_xlabel('Task Weight', fontsize=9)
    ax2.set_ylabel('Accuracy (%)', fontsize=9)
    ax2.set_title('(b) Prediction Accuracy by Task Weight', fontsize=10, fontweight='bold', pad=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_ylim([0, 105])
    
    for bar, acc, count in zip(bars, accuracy_by_weight, count_by_weight):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%\n(n={count})',
                ha='center', va='bottom', fontsize=7)
    
    # (c) Understanding rate (% Yes) by Task Weight
    ax3 = fig.add_subplot(gs[1, 0])
    understanding_rate = []
    
    for tw, yes, no in zip(task_weights, yes_counts, no_counts):
        total = yes + no
        rate = (yes / total * 100) if total > 0 else 0
        understanding_rate.append(rate)
    
    bars = ax3.bar(range(len(task_weights)), understanding_rate,
                   color=COLORS['accent1'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xticks(range(len(task_weights)))
    ax3.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax3.set_xlabel('Task Weight', fontsize=9)
    ax3.set_ylabel('Understanding Rate (%)', fontsize=9)
    ax3.set_title('(c) Understanding Rate by Task Weight', fontsize=10, fontweight='bold', pad=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_ylim([0, 105])
    
    for bar, rate in zip(bars, understanding_rate):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=7)
    
    # (d) Correct/Incorrect predictions by Task Weight
    ax4 = fig.add_subplot(gs[1, 1])
    correct_by_weight = []
    incorrect_by_weight = []
    
    for tw in task_weights:
        tw_data = [q for q in p_data if q['task_weight'] == tw]
        correct = sum(1 for q in tw_data if q['predicted_goal'] == q['actual_goal'])
        total = len([q for q in tw_data if q['predicted_goal'] is not None])
        incorrect = total - correct
        correct_by_weight.append(correct)
        incorrect_by_weight.append(incorrect)
    
    x = np.arange(len(task_weights))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, correct_by_weight, width, label='Correct',
                    color=COLORS['success'], alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax4.bar(x + width/2, incorrect_by_weight, width, label='Incorrect',
                    color=COLORS['warning'], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax4.set_xlabel('Task Weight', fontsize=9)
    ax4.set_ylabel('Count', fontsize=9)
    ax4.set_title('(d) Prediction Results by Task Weight', fontsize=10, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax4.legend(fontsize=8, frameon=False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=7)
    
    # (e) Response time by Task Weight
    ax5 = fig.add_subplot(gs[2, 0])
    time_by_weight = []
    
    for tw in task_weights:
        tw_data = [q for q in p_data if q['task_weight'] == tw]
        times = [q['timestamp'] for q in tw_data if q['timestamp'] > 0]
        avg_time = np.mean(times) if times else 0
        time_by_weight.append(avg_time)
    
    bars = ax5.bar(range(len(task_weights)), time_by_weight,
                   color=COLORS['neutral'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.set_xticks(range(len(task_weights)))
    ax5.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax5.set_xlabel('Task Weight', fontsize=9)
    ax5.set_ylabel('Avg Response Time (s)', fontsize=9)
    ax5.set_title('(e) Average Response Time by Task Weight', fontsize=10, fontweight='bold', pad=10)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    for bar, time in zip(bars, time_by_weight):
        height = bar.get_height()
        if height > 0:
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}s',
                    ha='center', va='bottom', fontsize=7)
    
    # (f) Statistics table by Task Weight
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    stats_data = [['Task Weight', 'Total', 'Yes', 'No', 'Accuracy']]
    
    for i, tw in enumerate(task_weights):
        total = yes_counts[i] + no_counts[i]
        stats_data.append([
            f'{tw:.1f}',
            f'{total}',
            f'{yes_counts[i]}',
            f'{no_counts[i]}',
            f'{accuracy_by_weight[i]:.1f}%'
        ])
    
    table = ax6.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
    
    ax6.set_title('(f) Summary Statistics by Task Weight', fontsize=10, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle(f'Participant {participant_id} - Questionnaire Analysis by Task Weight',
                fontsize=12, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / f'participant_{participant_id}_questionnaire.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved participant {participant_id} questionnaire analysis to {output_file}")

def analyze_overall_questionnaire(questionnaire_data, output_dir='questionnaire_results'):
    """Create overall summary of questionnaire data by task weight"""
    
    if not questionnaire_data:
        print("No questionnaire data to analyze")
        return
    
    # Get unique task weights
    task_weights = sorted(set(q['task_weight'] for q in questionnaire_data))
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.95, top=0.93, bottom=0.05)
    
    # (a) Overall Understanding by Task Weight
    ax1 = fig.add_subplot(gs[0, 0])
    yes_counts = []
    no_counts = []
    
    for tw in task_weights:
        tw_data = [q for q in questionnaire_data if q['task_weight'] == tw]
        yes_count = sum(1 for q in tw_data if q['understood'].lower() == 'yes')
        no_count = sum(1 for q in tw_data if q['understood'].lower() == 'no')
        yes_counts.append(yes_count)
        no_counts.append(no_count)
    
    x = np.arange(len(task_weights))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, yes_counts, width, label='Yes',
                    color=COLORS['success'], alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, no_counts, width, label='No',
                    color=COLORS['warning'], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Task Weight', fontsize=9)
    ax1.set_ylabel('Count', fontsize=9)
    ax1.set_title('(a) Understanding of Robot Intent by Task Weight (All Participants)', 
                  fontsize=10, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax1.legend(fontsize=8, frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=7)
    
    # (b) Prediction accuracy by Task Weight
    ax2 = fig.add_subplot(gs[0, 1])
    accuracy_by_weight = []
    count_by_weight = []
    
    for tw in task_weights:
        tw_data = [q for q in questionnaire_data if q['task_weight'] == tw]
        correct = sum(1 for q in tw_data if q['predicted_goal'] == q['actual_goal'])
        total = len([q for q in tw_data if q['predicted_goal'] is not None])
        acc = (correct / total * 100) if total > 0 else 0
        accuracy_by_weight.append(acc)
        count_by_weight.append(total)
    
    bars = ax2.bar(range(len(task_weights)), accuracy_by_weight,
                   color=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(task_weights)))
    ax2.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax2.set_xlabel('Task Weight', fontsize=9)
    ax2.set_ylabel('Accuracy (%)', fontsize=9)
    ax2.set_title('(b) Prediction Accuracy by Task Weight', fontsize=10, fontweight='bold', pad=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_ylim([0, 105])
    
    for bar, acc, count in zip(bars, accuracy_by_weight, count_by_weight):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%\n(n={count})',
                ha='center', va='bottom', fontsize=7)
    
    # (c) Understanding rate by Task Weight
    ax3 = fig.add_subplot(gs[1, 0])
    understanding_rate = []
    
    for yes, no in zip(yes_counts, no_counts):
        total = yes + no
        rate = (yes / total * 100) if total > 0 else 0
        understanding_rate.append(rate)
    
    bars = ax3.bar(range(len(task_weights)), understanding_rate,
                   color=COLORS['accent1'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xticks(range(len(task_weights)))
    ax3.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax3.set_xlabel('Task Weight', fontsize=9)
    ax3.set_ylabel('Understanding Rate (%)', fontsize=9)
    ax3.set_title('(c) Understanding Rate by Task Weight', fontsize=10, fontweight='bold', pad=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_ylim([0, 105])
    
    for bar, rate in zip(bars, understanding_rate):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=7)
    
    # (d) Correct/Incorrect by Task Weight
    ax4 = fig.add_subplot(gs[1, 1])
    correct_by_weight = []
    incorrect_by_weight = []
    
    for tw in task_weights:
        tw_data = [q for q in questionnaire_data if q['task_weight'] == tw]
        correct = sum(1 for q in tw_data if q['predicted_goal'] == q['actual_goal'])
        total = len([q for q in tw_data if q['predicted_goal'] is not None])
        incorrect = total - correct
        correct_by_weight.append(correct)
        incorrect_by_weight.append(incorrect)
    
    x = np.arange(len(task_weights))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, correct_by_weight, width, label='Correct',
                    color=COLORS['success'], alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax4.bar(x + width/2, incorrect_by_weight, width, label='Incorrect',
                    color=COLORS['warning'], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax4.set_xlabel('Task Weight', fontsize=9)
    ax4.set_ylabel('Count', fontsize=9)
    ax4.set_title('(d) Prediction Results by Task Weight', fontsize=10, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax4.legend(fontsize=8, frameon=False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=7)
    
    # (e) Response time by Task Weight
    ax5 = fig.add_subplot(gs[2, 0])
    time_by_weight = []
    
    for tw in task_weights:
        tw_data = [q for q in questionnaire_data if q['task_weight'] == tw]
        times = [q['timestamp'] for q in tw_data if q['timestamp'] > 0]
        avg_time = np.mean(times) if times else 0
        time_by_weight.append(avg_time)
    
    bars = ax5.bar(range(len(task_weights)), time_by_weight,
                   color=COLORS['neutral'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.set_xticks(range(len(task_weights)))
    ax5.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax5.set_xlabel('Task Weight', fontsize=9)
    ax5.set_ylabel('Avg Response Time (s)', fontsize=9)
    ax5.set_title('(e) Average Response Time by Task Weight', fontsize=10, fontweight='bold', pad=10)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    for bar, time in zip(bars, time_by_weight):
        height = bar.get_height()
        if height > 0:
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}s',
                    ha='center', va='bottom', fontsize=7)
    
    # (f) Statistics table by Task Weight
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    stats_data = [['Task Weight', 'Total', 'Yes (%)', 'No (%)', 'Accuracy']]
    
    for i, tw in enumerate(task_weights):
        total = yes_counts[i] + no_counts[i]
        yes_pct = (yes_counts[i] / total * 100) if total > 0 else 0
        no_pct = (no_counts[i] / total * 100) if total > 0 else 0
        stats_data.append([
            f'{tw:.1f}',
            f'{total}',
            f'{yes_counts[i]} ({yes_pct:.0f}%)',
            f'{no_counts[i]} ({no_pct:.0f}%)',
            f'{accuracy_by_weight[i]:.1f}%'
        ])
    
    # Add overall row
    total_all = sum(yes_counts) + sum(no_counts)
    yes_all = sum(yes_counts)
    no_all = sum(no_counts)
    yes_pct_all = (yes_all / total_all * 100) if total_all > 0 else 0
    no_pct_all = (no_all / total_all * 100) if total_all > 0 else 0
    overall_acc = np.mean(accuracy_by_weight)
    
    stats_data.append([
        'Overall',
        f'{total_all}',
        f'{yes_all} ({yes_pct_all:.0f}%)',
        f'{no_all} ({no_pct_all:.0f}%)',
        f'{overall_acc:.1f}%'
    ])
    
    table = ax6.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.8)
    
    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors and highlight overall row
    for i in range(1, len(stats_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i == len(stats_data) - 1:  # Overall row
                cell.set_facecolor(COLORS['accent1'])
                cell.set_text_props(weight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
    
    ax6.set_title('(f) Summary Statistics by Task Weight', fontsize=10, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle('Overall Questionnaire Analysis by Task Weight',
                fontsize=12, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'questionnaire_overall_summary.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved overall questionnaire analysis to {output_file}")

def main():
    """Main analysis function"""
    
    print("="*60)
    print("QUESTIONNAIRE DATA ANALYSIS (BY TASK WEIGHT)")
    print("="*60)
    
    # Load data
    print("\n1. Loading experiment data...")
    all_data = load_experiment_data()
    print(f"   Loaded data for {len(all_data)} participants")
    
    # Extract questionnaire data
    print("\n2. Extracting questionnaire responses...")
    questionnaire_data = extract_questionnaire_data(all_data)
    print(f"   Found {len(questionnaire_data)} questionnaire responses")
    
    if not questionnaire_data:
        print("\n   No questionnaire data found!")
        return
    
    # Print sample grouped by task weight
    print("\n3. Sample questionnaire data by task weight:")
    task_weights = sorted(set(q['task_weight'] for q in questionnaire_data))
    for tw in task_weights:
        tw_data = [q for q in questionnaire_data if q['task_weight'] == tw]
        yes_count = sum(1 for q in tw_data if q['understood'].lower() == 'yes')
        no_count = sum(1 for q in tw_data if q['understood'].lower() == 'no')
        print(f"   Task Weight {tw:.1f}: {len(tw_data)} responses ({yes_count} Yes, {no_count} No)")
    
    # Analyze each participant
    print("\n4. Generating individual participant analyses...")
    participant_ids = sorted(set(q['participant_id'] for q in questionnaire_data))
    
    for pid in participant_ids:
        analyze_participant_questionnaire(pid, questionnaire_data)
    
    # Overall analysis
    print("\n5. Generating overall summary analysis...")
    analyze_overall_questionnaire(questionnaire_data)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved in: questionnaire_results/")
    print(f"  - Individual analyses: participant_*_questionnaire.pdf")
    print(f"  - Overall summary: questionnaire_overall_summary.pdf")

if __name__ == '__main__':
    main()