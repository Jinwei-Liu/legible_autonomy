#!/usr/bin/env python3

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def setup_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
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
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.dpi': 300,
    })


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
        with open(filepath, 'r', encoding='utf-8') as f:
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
            
            # 只保留最完整的文件（优先选择有phase2数据的）
            if pid not in participants:
                participants[pid] = {
                    'data': data,
                    'phase2_trials': phase2_trials,
                    'total_trials': total_trials,
                    'file_size': file_size
                }
                print(f"  Participant {pid}: {phase1_trials} phase1, {phase2_trials} phase2 trials")
            else:
                stored = participants[pid]
                # 优先选择phase2数据更多的，其次选择总trials更多的
                if (phase2_trials > stored['phase2_trials'] or 
                    (phase2_trials == stored['phase2_trials'] and total_trials > stored['total_trials']) or
                    (phase2_trials == stored['phase2_trials'] and total_trials == stored['total_trials'] and file_size > stored['file_size'])):
                    participants[pid]['data'] = data
                    participants[pid]['phase2_trials'] = phase2_trials
                    participants[pid]['total_trials'] = total_trials
                    participants[pid]['file_size'] = file_size
                    print(f"  Participant {pid}: updated (more complete)")
    
    final_data = [p['data'] for p in participants.values()]
    print(f"[INFO] Loaded final data for {len(final_data)} unique participant(s)")
    
    return final_data


def extract_phase2_data(all_data):
    phase2_data = []
    
    for participant in all_data:
        participant_id = participant['participant_id']
        
        if 'phase2' not in participant:
            print(f"[WARNING] Participant {participant_id} has no phase2 data")
            continue
        
        phase2_trials = participant['phase2']['trials']
        
        for trial in phase2_trials:
            rating = trial.get('questionnaire', {})
            
            phase2_data.append({
                'participant_id': participant_id,
                'gender': participant.get('gender', 'unknown'),
                'age': participant.get('age', 0),
                'task_weight': trial['task_weight'],
                'target_goal_idx': trial['target_goal_idx'],
                'intuitiveness': rating.get('intuitiveness_score', 0),
                'collaboration': rating.get('collaboration_score', 0),
                'timestamp': rating.get('timestamp', ''),
                'duration': (
                    pd.to_datetime(trial.get('end_time', '')) - 
                    pd.to_datetime(trial.get('start_time', ''))
                ).total_seconds() if trial.get('end_time') and trial.get('start_time') else 0,
                'final_position': trial.get('final_position', [0, 0]),
                'num_frames': len(trial.get('frames', [])),
            })
    
    df = pd.DataFrame(phase2_data)
    
    print(f"\nData extraction complete:")
    print(f"  - Participants: {df['participant_id'].nunique()}")
    print(f"  - Total trials: {len(df)}")
    print(f"  - Task weights: {sorted(df['task_weight'].unique())}")
    print(f"  - Intuitiveness range: [{df['intuitiveness'].min()}, {df['intuitiveness'].max()}]")
    print(f"  - Collaboration range: [{df['collaboration'].min()}, {df['collaboration'].max()}]")
    
    return df


def compute_statistics(df):
    print("\n" + "="*70)
    print("  PHASE 2: SUBJECTIVE RATING ANALYSIS")
    print("="*70)
    
    rating_dimensions = ['intuitiveness', 'collaboration']
    
    print("\nDescriptive statistics by task weight:")
    print("-"*70)
    
    for weight in sorted(df['task_weight'].unique()):
        subset = df[df['task_weight'] == weight]
        print(f"\nTask Weight: {weight} (n={len(subset)})")
        
        for dim in rating_dimensions:
            values = subset[dim]
            print(f"  {dim.capitalize():15s}: M={values.mean():.2f}, SD={values.std():.2f}, "
                  f"Range=[{values.min():.1f}, {values.max():.1f}]")
    
    print("\n" + "-"*70)
    print("One-way ANOVA (Task Weight Effect)")
    print("-"*70)
    
    for dim in rating_dimensions:
        groups = [df[df['task_weight'] == w][dim].values 
                  for w in sorted(df['task_weight'].unique())]
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        print(f"\n{dim.capitalize()}:")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(f"  Significance: {sig}")
    
    print("\n" + "-"*70)
    print("Correlation analysis")
    print("-"*70)
    
    corr, p_val = stats.pearsonr(df['intuitiveness'], df['collaboration'])
    print(f"\nIntuitiveness vs Collaboration: r={corr:+.3f}, p={p_val:.4f}")
    
    print("\nTask Weight correlations:")
    for dim in rating_dimensions:
        corr, p_val = stats.pearsonr(df['task_weight'], df[dim])
        print(f"  {dim.capitalize():15s}: r={corr:+.3f}, p={p_val:.4f}")


def plot_ratings_by_task_weight(df, output_dir):
    setup_nature_style()
    
    rating_dimensions = ['intuitiveness', 'collaboration']
    task_weights = sorted(df['task_weight'].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#3C5488', '#F39B7F']
    
    for idx, dim in enumerate(rating_dimensions):
        ax = axes[idx]
        
        for i, weight in enumerate(task_weights):
            subset = df[df['task_weight'] == weight]
            values = subset[dim].values
            
            x_positions = np.random.normal(weight, 0.02, size=len(values))
            
            ax.scatter(x_positions, values, 
                      alpha=0.6, s=80, 
                      color=colors[idx], 
                      edgecolors='black', 
                      linewidth=0.5)
        
        means = [df[df['task_weight'] == w][dim].mean() for w in task_weights]
        ax.plot(task_weights, means, 
               color='red', linewidth=2.5, 
               marker='D', markersize=8, 
               label='Mean', zorder=10)
        
        ax.set_xlabel('Task Weight', fontsize=11)
        ax.set_ylabel(f'{dim.capitalize()} Score', fontsize=11)
        ax.set_title(f'({chr(97+idx)}) {dim.capitalize()}', 
                    fontsize=12, fontweight='normal')
        ax.set_ylim(0, 10.5)
        ax.set_xticks(task_weights)
        ax.axhline(y=5, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.legend(loc='best', frameon=False)
        ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'phase2_ratings_by_task_weight.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {filename}")
    plt.close()


def plot_individual_profiles(df, output_dir):
    setup_nature_style()
    
    rating_dimensions = ['intuitiveness', 'collaboration']
    task_weights = sorted(df['task_weight'].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#3C5488', '#F39B7F']
    
    for idx, dim in enumerate(rating_dimensions):
        ax = axes[idx]
        
        for participant_id in df['participant_id'].unique():
            participant_data = df[df['participant_id'] == participant_id]
            
            ratings = []
            for weight in task_weights:
                rating = participant_data[participant_data['task_weight'] == weight][dim]
                if len(rating) > 0:
                    ratings.append(rating.values[0])
                else:
                    ratings.append(np.nan)
            
            ax.plot(task_weights, ratings, 
                   alpha=0.4, linewidth=1.5, 
                   marker='o', markersize=5, 
                   color='gray',
                   label='Participants' if participant_id == df['participant_id'].unique()[0] else '')
        
        means = [df[df['task_weight'] == w][dim].mean() for w in task_weights]
        ax.plot(task_weights, means, 
               color=colors[idx], 
               linewidth=3, marker='o', markersize=10, 
               label='Mean', zorder=10)
        
        ax.set_xlabel('Task Weight', fontsize=11)
        ax.set_ylabel(f'{dim.capitalize()} Score', fontsize=11)
        ax.set_title(f'({chr(97+idx)}) {dim.capitalize()}', 
                    fontsize=12, fontweight='normal')
        ax.set_ylim(0, 10.5)
        ax.set_xticks(task_weights)
        ax.axhline(y=5, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.legend(loc='best', frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'phase2_individual_profiles.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {filename}")
    plt.close()


def plot_correlation(df, output_dir):
    setup_nature_style()
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    x = df['intuitiveness'].values
    y = df['collaboration'].values
    
    scatter = ax.scatter(x, y, 
                        c=df['task_weight'].values,
                        cmap='viridis',
                        s=100, alpha=0.6,
                        edgecolors='black', linewidth=0.5)
    
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
    
    corr, p_val = stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.4f}',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)
    
    ax.set_xlabel('Intuitiveness Score', fontsize=11)
    ax.set_ylabel('Collaboration Score', fontsize=11)
    ax.set_title('Correlation between Rating Dimensions', fontsize=12)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10.5)
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, linewidth=0.8)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Task Weight', rotation=270, labelpad=20)
    
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'phase2_correlation.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {filename}")
    plt.close()


def main():
    data_dir = 'experiment_data'
    output_dir = 'analysis_results/phase2'
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' not found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("  PHASE 2 ANALYSIS")
    print("="*70)
    
    print("\n1. Loading experiment data...")
    all_data = load_all_experiment_data(data_dir)
    
    if len(all_data) == 0:
        print("[ERROR] No data files found!")
        return
    
    print("\n2. Extracting Phase 2 data...")
    df = extract_phase2_data(all_data)
    
    if len(df) == 0:
        print("[ERROR] No Phase 2 data found!")
        return
    
    print("\n3. Statistical analysis...")
    compute_statistics(df)
    
    print("\n4. Generating visualizations...")
    plot_ratings_by_task_weight(df, output_dir)
    plot_individual_profiles(df, output_dir)
    plot_correlation(df, output_dir)
    
    print("\n5. Exporting CSV...")
    csv_filename = os.path.join(output_dir, 'phase2_data.csv')
    df.to_csv(csv_filename, index=False)
    print(f"[SAVED] {csv_filename}")
    
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - phase2_ratings_by_task_weight.png")
    print("  - phase2_individual_profiles.png")
    print("  - phase2_correlation.png")
    print("  - phase2_data.csv")


if __name__ == '__main__':
    main()