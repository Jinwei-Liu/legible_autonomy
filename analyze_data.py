#!/usr/bin/env python3
"""
Combined Figure Analysis with Statistical Tests
Generates visualization (a)(b)(c)(d) and performs hypothesis testing
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, friedmanchisquare, spearmanr, pearsonr, mannwhitneyu, kruskal
import os
from pathlib import Path
from itertools import combinations
from datetime import datetime
import glob

def setup_plot_style(font_size=20, label_size=22, tick_size=18,
                     line_width=0.8, tick_width=0.8, tick_length=4):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = label_size
    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['ytick.labelsize'] = tick_size
    plt.rcParams['axes.linewidth'] = line_width
    plt.rcParams['xtick.major.width'] = tick_width
    plt.rcParams['ytick.major.width'] = tick_width
    plt.rcParams['xtick.major.size'] = tick_length
    plt.rcParams['ytick.major.size'] = tick_length
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

COLORS = {
    'success': '#06A77D',
    'warning': '#F77F00',
}

TASK_WEIGHT_COLORS = {
    0:  '#E64B35',
    5:  '#4DBBD5',
    10: '#00A087',
}
TASK_WEIGHT_COLORS_LIST = ['#E64B35', '#4DBBD5', '#00A087']

# ==================== DATA LOADING ====================

def load_experiment_data(data_dir='experiment_data'):
    data_path = Path(data_dir)
    all_data = []
    participant_files = {}

    for json_file in data_path.glob('participant_*.json'):
        parts = json_file.stem.split('_')
        if len(parts) >= 2:
            pid = parts[1]
            if pid not in participant_files:
                participant_files[pid] = []
            participant_files[pid].append(json_file)

    for pid, files in participant_files.items():
        latest_file = max(files, key=lambda f: f.stat().st_size)
        with open(latest_file, 'r') as f:
            all_data.append(json.load(f))

    return all_data


def extract_questionnaire_data(all_data):
    questionnaire_data = []
    for participant in all_data:
        participant_id = participant['participant_id']
        trials = participant.get('phase1', participant).get('trials', [])

        for trial in trials:
            if 'questionnaire' in trial:
                q_data = trial['questionnaire']
                questionnaire_data.append({
                    'participant_id': participant_id,
                    'trial_number': trial['trial_num'],
                    'task_weight': trial['task_weight'],
                    'actual_goal': trial['target_goal'],
                    'understood': q_data.get('understood', 'unknown'),
                    'predicted_goal': q_data.get('predicted_goal', None),
                })
    return questionnaire_data


def extract_phase2_data(all_data):
    phase2_data = []
    for participant in all_data:
        if 'phase2' not in participant:
            continue

        for trial in participant['phase2']['trials']:
            rating = trial.get('questionnaire', {})
            phase2_data.append({
                'participant_id': participant['participant_id'],
                'task_weight': trial['task_weight'],
                'intuitiveness': rating.get('intuitiveness_score', 0),
                'collaboration': rating.get('collaboration_score', 0),
            })

    return pd.DataFrame(phase2_data)


def extract_trial_metrics(trial):
    """直接复制自 analyze_experiment_data.py"""
    frames = trial['frames']
    positions     = np.array([f['position']     for f in frames])
    user_inputs   = np.array([f['user_input']   for f in frames])
    robot_actions = np.array([f['robot_action'] for f in frames])
    beliefs       = np.array([f['beliefs']      for f in frames])
    betas         = np.array([f['beta']         for f in frames])

    times      = [datetime.fromisoformat(f['time']) for f in frames]
    start_time = times[0]
    time_series = [(t - start_time).total_seconds() for t in times]

    return {
        'task_weight':          trial['task_weight'],
        'avg_user_input_norm':  float(np.mean(np.linalg.norm(user_inputs,   axis=1))),
        'avg_robot_action_norm':float(np.mean(np.linalg.norm(robot_actions, axis=1))),
        'duration':             time_series[-1] if time_series else 0,
    }


def extract_user_input_data(all_data):
    """收集所有 trial 的 avg_user_input_norm，按 task_weight 分组"""
    values_by_weight = {}
    for participant in all_data:
        if 'phase1' in participant:
            trials = participant['phase1']['trials']
        elif 'trials' in participant:
            trials = participant['trials']
        else:
            continue

        for trial in trials:
            try:
                m = extract_trial_metrics(trial)
                tw = m['task_weight']
                if tw not in values_by_weight:
                    values_by_weight[tw] = []
                values_by_weight[tw].append(m['avg_user_input_norm'])
            except Exception:
                continue

    return {tw: np.array(v) for tw, v in values_by_weight.items()}


# ==================== STATISTICAL TESTS ====================

def sig_marker(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def test_understanding_by_task_weight(questionnaire_data):
    print("\n" + "="*70)
    print("  UNDERSTANDING RATE ANALYSIS")
    print("="*70)

    task_weights = sorted(set([q['task_weight'] for q in questionnaire_data]))
    contingency_table = []
    for tw in task_weights:
        tw_data = [q for q in questionnaire_data if q['task_weight'] == tw]
        yes_count = sum(1 for q in tw_data if q['understood'].lower() == 'yes')
        no_count  = sum(1 for q in tw_data if q['understood'].lower() == 'no')
        contingency_table.append([yes_count, no_count])
        print(f"\nTask Weight {tw}:")
        print(f"  Yes: {yes_count} ({yes_count/(yes_count+no_count)*100:.1f}%)")
        print(f"  No:  {no_count} ({no_count/(yes_count+no_count)*100:.1f}%)")

    contingency_table = np.array(contingency_table)
    chi2, p_value, dof, _ = chi2_contingency(contingency_table)
    print(f"\nChi-square: χ²={chi2:.4f}, df={dof}, p={p_value:.4f} {sig_marker(p_value)}")

    for i, j in combinations(range(len(task_weights)), 2):
        tw1, tw2 = task_weights[i], task_weights[j]
        table_2x2 = contingency_table[[i, j], :]
        chi2_pair, p_pair, _, _ = chi2_contingency(table_2x2)
        n_comp = len(list(combinations(range(len(task_weights)), 2)))
        p_corr = min(p_pair * n_comp, 1.0)
        print(f"  {tw1} vs {tw2}: p={p_pair:.4f}, p_corrected={p_corr:.4f} {sig_marker(p_corr)}")

    return {'chi2': chi2, 'p_value': p_value, 'dof': dof, 'contingency_table': contingency_table}


def test_prediction_accuracy_by_task_weight(questionnaire_data):
    print("\n" + "="*70)
    print("  PREDICTION ACCURACY ANALYSIS")
    print("="*70)

    task_weights = sorted(set([q['task_weight'] for q in questionnaire_data]))
    contingency_table = []
    for tw in task_weights:
        tw_data = [q for q in questionnaire_data if q['task_weight'] == tw]
        correct = sum(1 for q in tw_data if q['predicted_goal'] == q['actual_goal'])
        total   = len([q for q in tw_data if q['predicted_goal'] is not None])
        contingency_table.append([correct, total - correct])
        print(f"\nTask Weight {tw}: correct={correct}/{total} ({correct/total*100:.1f}%)")

    contingency_table = np.array(contingency_table)
    chi2, p_value, dof, _ = chi2_contingency(contingency_table)
    print(f"\nChi-square: χ²={chi2:.4f}, df={dof}, p={p_value:.4f} {sig_marker(p_value)}")

    for i, j in combinations(range(len(task_weights)), 2):
        tw1, tw2 = task_weights[i], task_weights[j]
        table_2x2 = contingency_table[[i, j], :]
        chi2_pair, p_pair, _, _ = chi2_contingency(table_2x2)
        n_comp = len(list(combinations(range(len(task_weights)), 2)))
        p_corr = min(p_pair * n_comp, 1.0)
        print(f"  {tw1} vs {tw2}: p={p_pair:.4f}, p_corrected={p_corr:.4f} {sig_marker(p_corr)}")

    return {'chi2': chi2, 'p_value': p_value, 'dof': dof, 'contingency_table': contingency_table}


def test_phase2_ratings(phase2_df):
    print("\n" + "="*70)
    print("  PHASE 2 SUBJECTIVE RATINGS ANALYSIS")
    print("="*70)

    if len(phase2_df) == 0:
        print("\nNo Phase 2 data found")
        return None

    task_weights = sorted(phase2_df['task_weight'].unique())

    print("\n--- INTUITIVENESS SCORES ---")
    participant_data = {}
    for _, row in phase2_df.iterrows():
        pid, tw = row['participant_id'], row['task_weight']
        if pid not in participant_data:
            participant_data[pid] = {}
        participant_data[pid][tw] = row['intuitiveness']

    complete = [pid for pid, d in participant_data.items() if len(d) == len(task_weights)]
    intuit_by_tw = {}
    for tw in task_weights:
        scores = phase2_df[phase2_df['task_weight'] == tw]['intuitiveness'].values
        intuit_by_tw[tw] = scores
        print(f"Task Weight {tw}: Mean={np.mean(scores):.2f}, SD={np.std(scores, ddof=1):.2f}, N={len(scores)}")

    if len(complete) > 0:
        arrays = [[participant_data[pid][tw] for pid in complete] for tw in task_weights]
        stat, p = friedmanchisquare(*arrays)
        print(f"\nFriedman test (N={len(complete)}): χ²={stat:.4f}, p={p:.4f} {sig_marker(p)}")

    print("\n--- COLLABORATION SCORES ---")
    participant_data_c = {}
    for _, row in phase2_df.iterrows():
        pid, tw = row['participant_id'], row['task_weight']
        if pid not in participant_data_c:
            participant_data_c[pid] = {}
        participant_data_c[pid][tw] = row['collaboration']

    complete_c = [pid for pid, d in participant_data_c.items() if len(d) == len(task_weights)]
    collab_by_tw = {}
    for tw in task_weights:
        scores = phase2_df[phase2_df['task_weight'] == tw]['collaboration'].values
        collab_by_tw[tw] = scores
        print(f"Task Weight {tw}: Mean={np.mean(scores):.2f}, SD={np.std(scores, ddof=1):.2f}, N={len(scores)}")

    if len(complete_c) > 0:
        arrays_c = [[participant_data_c[pid][tw] for pid in complete_c] for tw in task_weights]
        stat_c, p_c = friedmanchisquare(*arrays_c)
        print(f"\nFriedman test (N={len(complete_c)}): χ²={stat_c:.4f}, p={p_c:.4f} {sig_marker(p_c)}")

    print("\n--- CORRELATION ANALYSIS ---")
    intuit_all = phase2_df['intuitiveness'].values
    collab_all = phase2_df['collaboration'].values
    r_p, p_p = pearsonr(intuit_all, collab_all)
    r_s, p_s = spearmanr(intuit_all, collab_all)
    print(f"Pearson:  r={r_p:.4f}, p={p_p:.4f} {sig_marker(p_p)}")
    print(f"Spearman: ρ={r_s:.4f}, p={p_s:.4f} {sig_marker(p_s)}")

    return {'intuitiveness_by_tw': intuit_by_tw, 'collaboration_by_tw': collab_by_tw,
            'correlation': {'pearson_r': r_p, 'pearson_p': p_p, 'spearman_r': r_s, 'spearman_p': p_s}}


def test_user_input_norm(values_by_weight):
    """
    Kruskal-Wallis (overall) + pairwise Mann-Whitney U (Bonferroni) for panel (d)
    Returns dict of pairwise results for significance bar drawing
    """
    print("\n" + "="*70)
    print("  USER INPUT NORM ANALYSIS (Panel d)")
    print("="*70)

    task_weights = sorted(values_by_weight.keys())
    arrays = [values_by_weight[tw] for tw in task_weights]

    stat, p = kruskal(*arrays)
    print(f"\nKruskal-Wallis: H={stat:.4f}, p={p:.4f} {sig_marker(p)}")

    n_comp = len(list(combinations(task_weights, 2)))
    pairwise = {}
    for tw1, tw2 in combinations(task_weights, 2):
        u, p_raw = mannwhitneyu(values_by_weight[tw1], values_by_weight[tw2],
                                alternative='two-sided')
        p_corr = min(p_raw * n_comp, 1.0)
        marker = sig_marker(p_corr)
        pairwise[(tw1, tw2)] = {'u': u, 'p_raw': p_raw, 'p_corr': p_corr, 'marker': marker}
        print(f"  {tw1} vs {tw2}: U={u:.1f}, p={p_raw:.4f}, p_corrected={p_corr:.4f} {marker}")

    return {'kruskal_stat': stat, 'kruskal_p': p, 'pairwise': pairwise}


# ==================== VISUALIZATION ====================

def plot_understanding_subplot(ax, questionnaire_data, text_size=14):
    task_weights = sorted(set([q['task_weight'] for q in questionnaire_data]))
    yes_pcts, no_pcts, yes_counts, no_counts = [], [], [], []

    for tw in task_weights:
        tw_data = [q for q in questionnaire_data if q['task_weight'] == tw]
        yes_count = sum(1 for q in tw_data if q['understood'].lower() == 'yes')
        no_count  = sum(1 for q in tw_data if q['understood'].lower() == 'no')
        total = yes_count + no_count
        yes_pcts.append((yes_count / total * 100) if total > 0 else 0)
        no_pcts.append((no_count  / total * 100) if total > 0 else 0)
        yes_counts.append(yes_count)
        no_counts.append(no_count)

    y = np.arange(len(task_weights))
    bars1 = ax.barh(y, yes_pcts, 0.6, label='Yes (Understood)',
                    color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax.barh(y, no_pcts,  0.6, left=yes_pcts, label='No (Not Understood)',
                    color=COLORS['warning'], alpha=0.8, edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Task Weight')
    ax.set_xlabel('Percentage (%)')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{tw:.1f}' for tw in task_weights])
    ax.set_xlim(0, 105)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

    for bar1, bar2, yes_pct, no_pct, yes_cnt, no_cnt in zip(
            bars1, bars2, yes_pcts, no_pcts, yes_counts, no_counts):
        if yes_pct > 10:
            ax.text(yes_pct / 2, bar1.get_y() + bar1.get_height() / 2,
                    f'{yes_pct:.1f}%\n(n={yes_cnt})',
                    ha='center', va='center', fontsize=text_size, fontweight='bold', color='white')
        if no_pct > 10:
            ax.text(yes_pct + no_pct / 2, bar2.get_y() + bar2.get_height() / 2,
                    f'{no_pct:.1f}%\n(n={no_cnt})',
                    ha='center', va='center', fontsize=text_size, fontweight='bold', color='white')
        ax.text(103, bar2.get_y() + bar2.get_height() / 2,
                f'N={yes_cnt + no_cnt}',
                ha='left', va='center', fontsize=text_size - 2, color='gray')


def plot_prediction_subplot(ax, questionnaire_data, text_size=14):
    task_weights = sorted(set([q['task_weight'] for q in questionnaire_data]))
    correct_pcts, incorrect_pcts, correct_counts, incorrect_counts = [], [], [], []

    for tw in task_weights:
        tw_data = [q for q in questionnaire_data if q['task_weight'] == tw]
        correct = sum(1 for q in tw_data if q['predicted_goal'] == q['actual_goal'])
        total   = len([q for q in tw_data if q['predicted_goal'] is not None])
        incorrect = total - correct
        correct_pcts.append((correct / total * 100) if total > 0 else 0)
        incorrect_pcts.append((incorrect / total * 100) if total > 0 else 0)
        correct_counts.append(correct)
        incorrect_counts.append(incorrect)

    y = np.arange(len(task_weights))
    bars1 = ax.barh(y, correct_pcts, 0.6, label='Correct',
                    color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax.barh(y, incorrect_pcts, 0.6, left=correct_pcts, label='Incorrect',
                    color=COLORS['warning'], alpha=0.8, edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Task Weight')
    ax.set_xlabel('Percentage (%)')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{tw:.1f}' for tw in task_weights])
    ax.set_xlim(0, 105)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

    for bar1, bar2, correct_pct, incorrect_pct, correct_cnt, incorrect_cnt in zip(
            bars1, bars2, correct_pcts, incorrect_pcts, correct_counts, incorrect_counts):
        if correct_pct > 10:
            ax.text(correct_pct / 2, bar1.get_y() + bar1.get_height() / 2,
                    f'{correct_pct:.1f}%\n(n={correct_cnt})',
                    ha='center', va='center', fontsize=text_size, fontweight='bold', color='white')
        if incorrect_pct > 10:
            ax.text(correct_pct + incorrect_pct / 2, bar2.get_y() + bar2.get_height() / 2,
                    f'{incorrect_pct:.1f}%\n(n={incorrect_cnt})',
                    ha='center', va='center', fontsize=text_size, fontweight='bold', color='white')
        ax.text(103, bar2.get_y() + bar2.get_height() / 2,
                f'N={correct_cnt + incorrect_cnt}',
                ha='left', va='center', fontsize=text_size - 2, color='gray')


def plot_correlation_subplot(ax, df):
    x = df['intuitiveness'].values
    y = df['collaboration'].values
    task_weights = df['task_weight'].values
    unique_weights = sorted(df['task_weight'].unique())

    for tw in unique_weights:
        mask = task_weights == tw
        color = TASK_WEIGHT_COLORS.get(tw, '#999999')
        ax.scatter(x[mask], y[mask], c=color, s=100, alpha=0.6,
                   edgecolors='black', linewidth=0.8, label=f'Task Weight {tw:.1f}')

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2.5, alpha=0.8, label='Linear fit')

    ax.set_xlabel('Intuitiveness Score')
    ax.set_ylabel('Collaboration Score')
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10.5)
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, linewidth=1)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)


def add_significance_bars(ax, x_positions, values_by_weight, task_weights,
                          pairwise_results, tick_size=18):
    """
    在箱线图上方绘制显著性横线 + * 标注。
    只绘制有显著差异的组对（marker != 'ns'）。
    自动按组对间距叠放，避免重叠。
    """
    sig_pairs = [(tw1, tw2) for (tw1, tw2), r in pairwise_results.items()
                 if r['marker'] != 'ns']

    if not sig_pairs:
        return

    all_vals = np.concatenate([values_by_weight[tw] for tw in task_weights])
    y_max_data = all_vals.max()
    y_span = y_max_data

    bar_height   = y_span * 0.06   # 竖线高度
    bar_gap      = y_span * 0.10   # 每层横线间距
    base_y       = y_max_data + y_span * 0.08  # 第一层起始 y

    tw_to_x = {tw: x_positions[i] for i, tw in enumerate(task_weights)}

    for level, (tw1, tw2) in enumerate(sig_pairs):
        marker = pairwise_results[(tw1, tw2)]['marker']
        x1, x2 = tw_to_x[tw1], tw_to_x[tw2]
        y = base_y + level * bar_gap

        # 横线
        ax.plot([x1, x2], [y, y], color='black', linewidth=1.2)
        # 两端竖线
        ax.plot([x1, x1], [y - bar_height * 0.4, y], color='black', linewidth=1.2)
        ax.plot([x2, x2], [y - bar_height * 0.4, y], color='black', linewidth=1.2)
        # 显著性标记
        ax.text((x1 + x2) / 2, y + y_span * 0.01, marker,
                ha='center', va='bottom', fontsize=tick_size, color='black')

    # 更新 y 轴上限以容纳标注
    n_levels = len(sig_pairs)
    new_top = base_y + n_levels * bar_gap + y_span * 0.15
    current_bottom = ax.get_ylim()[0]
    ax.set_ylim(current_bottom, new_top)


def plot_user_input_subplot(ax, values_by_weight, pairwise_results,
                            tick_size=18, label_size=20):
    """
    Panel (d)：箱线图 + jitter + 均值菱形 + 显著性标注
    风格与 (a)(b)(c) 完全一致（无标题，字号统一）
    """
    task_weights = sorted(values_by_weight.keys())
    n = len(task_weights)
    colors = [TASK_WEIGHT_COLORS.get(tw, '#999999') for tw in task_weights]
    data_list = [values_by_weight[tw] for tw in task_weights]
    means = [np.mean(v) for v in data_list]
    x = np.arange(n)
    box_width = 0.45

    # 1. 箱线图
    bp = ax.boxplot(data_list,
                    positions=x,
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2, linestyle='--', color='#555555'),
                    capprops=dict(linewidth=1.2, color='#555555'),
                    boxprops=dict(linewidth=1.2),
                    zorder=2)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # 2. Jitter strip
    rng = np.random.default_rng(42)
    for i, (tw, color) in enumerate(zip(task_weights, colors)):
        vals = values_by_weight[tw]
        jitter = rng.uniform(-box_width * 0.28, box_width * 0.28, size=len(vals))
        ax.scatter(i + jitter, vals,
                   color=color, edgecolors='white', linewidths=0.4,
                   alpha=0.55, s=28, zorder=3)

    # 3. 均值菱形
    ax.scatter(x, means,
               marker='D', color='white', edgecolors='black',
               linewidths=1.2, s=55, zorder=5, label='Mean')

    # 4. 显著性横线（调用统一函数）
    add_significance_bars(ax, x, values_by_weight, task_weights,
                          pairwise_results, tick_size=tick_size)

    # 轴标签（字号与 abc 一致）
    ax.set_xticks(x)
    ax.set_xticklabels([f'{tw:.1f}' for tw in task_weights])
    ax.set_xlabel('Task Weight', fontsize=label_size)
    ax.set_ylabel('User Effort', fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ==================== COMBINED FIGURE ====================

def plot_combined_figure(questionnaire_data, phase2_df, values_by_weight,
                         pairwise_results, output_dir,
                         font_size=18, label_size=20, tick_size=16, text_size=14):
    setup_plot_style(font_size=font_size, label_size=label_size, tick_size=tick_size)

    # 4 个子图横排，d 图稍窄一点
    fig = plt.figure(figsize=(26, 4))
    gs = fig.add_gridspec(1, 4, wspace=0.5, width_ratios=[1, 1, 1, 0.85])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    plot_understanding_subplot(ax1, questionnaire_data, text_size)
    plot_prediction_subplot(ax2, questionnaire_data, text_size)

    if len(phase2_df) > 0:
        plot_correlation_subplot(ax3, phase2_df)

    plot_user_input_subplot(ax4, values_by_weight, pairwise_results,
                            tick_size=tick_size, label_size=label_size)

    # 面板标签 (a)(b)(c)(d)
    for ax, label in zip([ax1, ax2, ax3, ax4], ['(a)', '(b)', '(c)', '(d)']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=label_size, fontweight='bold')

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'combined_figure.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] {output_file}")


# ==================== MAIN ====================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate combined analysis figure with statistical tests')
    parser.add_argument('--input',      '-i', type=str, default='./experiment_data')
    parser.add_argument('--output',     '-o', type=str, default='./figures')
    parser.add_argument('--font-size',        type=int, default=18)
    parser.add_argument('--label-size',       type=int, default=20)
    parser.add_argument('--tick-size',        type=int, default=16)
    parser.add_argument('--text-size',        type=int, default=14)
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  COMBINED FIGURE GENERATION WITH STATISTICAL ANALYSIS")
    print("="*70)

    all_data = load_experiment_data(args.input)
    if not all_data:
        print("[ERROR] No data found")
        return

    questionnaire_data = extract_questionnaire_data(all_data)
    phase2_df          = extract_phase2_data(all_data)
    values_by_weight   = extract_user_input_data(all_data)

    print(f"\nFound {len(all_data)} participants")
    print(f"Found {len(questionnaire_data)} questionnaire responses")
    print(f"Found {len(phase2_df)} phase2 ratings")
    print(f"Found user input data for task weights: {sorted(values_by_weight.keys())}")

    # 统计检验
    test_understanding_by_task_weight(questionnaire_data)
    test_prediction_accuracy_by_task_weight(questionnaire_data)
    test_phase2_ratings(phase2_df)
    user_input_results = test_user_input_norm(values_by_weight)

    # 生成图
    plot_combined_figure(
        questionnaire_data, phase2_df, values_by_weight,
        user_input_results['pairwise'],
        args.output,
        font_size=args.font_size,
        label_size=args.label_size,
        tick_size=args.tick_size,
        text_size=args.text_size,
    )

    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)
    print(f"Figure saved to: {args.output}/combined_figure.pdf")

if __name__ == '__main__':
    main()