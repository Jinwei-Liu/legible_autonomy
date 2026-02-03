#!/usr/bin/env python3
"""
修复版 Phase 2 数据分析脚本
根据实际收集的数据结构进行分析
"""

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
    """Setup matplotlib for Nature-style figures"""
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
    """加载所有实验数据文件"""
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    
    all_data = []
    for filepath in sorted(json_files):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['filepath'] = filepath
            all_data.append(data)
    
    print(f"[INFO] 加载了 {len(all_data)} 个实验数据文件")
    return all_data


def extract_phase2_data(all_data):
    """
    提取Phase 2的数据
    ⚠️ 修复: 使用 'questionnaire' 而不是 'subjective_rating'
    """
    phase2_data = []
    
    for participant in all_data:
        participant_id = participant['participant_id']
        
        if 'phase2' not in participant:
            print(f"[WARNING] Participant {participant_id} has no phase2 data")
            continue
        
        phase2_trials = participant['phase2']['trials']
        
        for trial in phase2_trials:
            # ✅ 修复: 从 'questionnaire' 而不是 'subjective_rating' 读取
            rating = trial.get('questionnaire', {})
            
            phase2_data.append({
                'participant_id': participant_id,
                'gender': participant.get('gender', 'unknown'),
                'age': participant.get('age', 0),
                'task_weight': trial['task_weight'],
                'target_goal_idx': trial['target_goal_idx'],
                # ✅ 修复: 使用实际存在的字段
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
    
    # 数据验证
    print(f"\n数据提取完成:")
    print(f"  - 参与者数量: {df['participant_id'].nunique()}")
    print(f"  - 总试次数: {len(df)}")
    print(f"  - Task weights: {sorted(df['task_weight'].unique())}")
    print(f"  - Intuitiveness评分范围: [{df['intuitiveness'].min()}, {df['intuitiveness'].max()}]")
    print(f"  - Collaboration评分范围: [{df['collaboration'].min()}, {df['collaboration'].max()}]")
    
    return df


def compute_statistics(df):
    """计算描述性统计和推断统计"""
    
    print("\n" + "="*70)
    print("  PHASE 2: 主观评分分析 (修复版)")
    print("="*70)
    
    # 评分维度 (实际收集的两个维度)
    rating_dimensions = ['intuitiveness', 'collaboration']
    
    # 按task weight分组的描述性统计
    print("\n按Task Weight分组的描述性统计:")
    print("-"*70)
    
    for weight in sorted(df['task_weight'].unique()):
        subset = df[df['task_weight'] == weight]
        print(f"\nTask Weight: {weight} (n={len(subset)})")
        
        for dim in rating_dimensions:
            values = subset[dim]
            print(f"  {dim.capitalize():15s}: M={values.mean():.2f}, SD={values.std():.2f}, "
                  f"Range=[{values.min():.1f}, {values.max():.1f}]")
    
    # ANOVA测试
    print("\n" + "-"*70)
    print("单因素方差分析 (Task Weight Effect)")
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
    
    # 相关性分析
    print("\n" + "-"*70)
    print("相关性分析")
    print("-"*70)
    
    # 两个维度之间的相关性
    corr, p_val = stats.pearsonr(df['intuitiveness'], df['collaboration'])
    print(f"\nIntuitiveness vs Collaboration: r={corr:+.3f}, p={p_val:.4f}")
    
    # Task weight与各维度的相关性
    print("\nTask Weight相关性:")
    for dim in rating_dimensions:
        corr, p_val = stats.pearsonr(df['task_weight'], df[dim])
        print(f"  {dim.capitalize():15s}: r={corr:+.3f}, p={p_val:.4f}")


def plot_ratings_by_task_weight(df, output_dir):
    """
    绘制按task weight分组的评分
    横轴: Task Weight
    纵轴: 每个参与者的评分值
    """
    setup_nature_style()
    
    rating_dimensions = ['intuitiveness', 'collaboration']
    task_weights = sorted(df['task_weight'].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#3C5488', '#F39B7F']  # 蓝色和橙色
    
    for idx, dim in enumerate(rating_dimensions):
        ax = axes[idx]
        
        # 为每个task weight绘制数据点
        for i, weight in enumerate(task_weights):
            subset = df[df['task_weight'] == weight]
            values = subset[dim].values
            
            # 添加抖动以避免重叠
            x_positions = np.random.normal(weight, 0.02, size=len(values))
            
            ax.scatter(x_positions, values, 
                      alpha=0.6, s=80, 
                      color=colors[idx], 
                      edgecolors='black', 
                      linewidth=0.5)
        
        # 绘制均值连线
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
    
    filename = os.path.join(output_dir, 'phase2_ratings_by_task_weight_FIXED.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {filename}")
    plt.close()


def plot_individual_profiles(df, output_dir):
    """
    绘制每个参与者的评分曲线
    横轴: Task Weight  
    纵轴: 评分值
    每条线代表一个参与者
    """
    setup_nature_style()
    
    rating_dimensions = ['intuitiveness', 'collaboration']
    task_weights = sorted(df['task_weight'].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#3C5488', '#F39B7F']
    
    for idx, dim in enumerate(rating_dimensions):
        ax = axes[idx]
        
        # 绘制每个参与者的曲线
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
        
        # 绘制平均曲线
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
    
    filename = os.path.join(output_dir, 'phase2_individual_profiles_FIXED.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {filename}")
    plt.close()


def plot_correlation(df, output_dir):
    """绘制两个维度之间的相关性"""
    setup_nature_style()
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    x = df['intuitiveness'].values
    y = df['collaboration'].values
    
    # 散点图
    scatter = ax.scatter(x, y, 
                        c=df['task_weight'].values,
                        cmap='viridis',
                        s=100, alpha=0.6,
                        edgecolors='black', linewidth=0.5)
    
    # 添加回归线
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
    
    # 相关系数
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
    
    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Task Weight', rotation=270, labelpad=20)
    
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'phase2_correlation_FIXED.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {filename}")
    plt.close()


def main():
    """主分析流程"""
    
    data_dir = 'experiment_data'
    output_dir = 'analysis_results/phase2_fixed'
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' not found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("  PHASE 2 分析 (修复版)")
    print("  使用实际收集的数据: intuitiveness & collaboration")
    print("="*70)
    
    # 1. 加载数据
    print("\n1. 加载实验数据...")
    all_data = load_all_experiment_data(data_dir)
    
    if len(all_data) == 0:
        print("[ERROR] 没有找到数据文件!")
        return
    
    # 2. 提取Phase 2数据
    print("\n2. 提取Phase 2数据...")
    df = extract_phase2_data(all_data)
    
    if len(df) == 0:
        print("[ERROR] 没有找到Phase 2数据!")
        return
    
    # 3. 统计分析
    print("\n3. 统计分析...")
    compute_statistics(df)
    
    # 4. 可视化
    print("\n4. 生成可视化...")
    plot_ratings_by_task_weight(df, output_dir)
    plot_individual_profiles(df, output_dir)
    plot_correlation(df, output_dir)
    
    # 5. 导出数据
    print("\n5. 导出CSV...")
    csv_filename = os.path.join(output_dir, 'phase2_data_FIXED.csv')
    df.to_csv(csv_filename, index=False)
    print(f"[SAVED] {csv_filename}")
    
    print("\n" + "="*70)
    print("  分析完成!")
    print("="*70)
    print(f"\n结果保存在: {output_dir}/")
    print("\n生成的文件:")
    print("  - phase2_ratings_by_task_weight_FIXED.png")
    print("  - phase2_individual_profiles_FIXED.png")
    print("  - phase2_correlation_FIXED.png")
    print("  - phase2_data_FIXED.csv")


if __name__ == '__main__':
    main()