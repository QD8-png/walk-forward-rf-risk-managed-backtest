# -*- coding: utf-8 -*-
"""
一键生成论文所有真图（图1, 图2, 图3, 图4, 图5）
数据来源：项目内的 CSV 文件
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径定义
git_base = r'C:\Users\qwe\.gemini\antigravity\scratch\walk-forward-rf-risk-managed-backtest'
plots_out = os.path.join(git_base, 'plots')
os.makedirs(plots_out, exist_ok=True)

# 1. 绘图数据读取
ablation_csv = os.path.join(git_base, 'sci_ablation_results_v2.csv')
yearly_csv = os.path.join(git_base, 'yearly_breakdown.csv')
equity_csv = os.path.join(git_base, 'equity_curves_117_pool.csv')
robust_csv = os.path.join(git_base, 'robustness_200_results.csv')

# -------------------------------------------------------------
# 图 1. 六阶段消融研究结果（累计收益与最大回撤双面板柱状图）
# -------------------------------------------------------------
def plot_figure_1():
    if not os.path.exists(ablation_csv):
        print("  跳过图1：sci_ablation_results_v2.csv 不存在")
        return
    
    df = pd.read_csv(ablation_csv, encoding='utf-8')
    
    # 转换为数值型
    df['Return_val'] = df['Return'].str.rstrip('%').astype(float)
    df['MDD_val'] = df['MDD'].str.rstrip('%').astype(float)
    
    systems = df['System'].tolist()
    # 缩短名称以便绘图
    labels = [s.split(':')[0] if ':' in s else s for s in systems]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    
    # Panel A: Return
    colors_ret = ['#E74C3C' if 'Pure Rules' in s else '#3498DB' for s in systems]
    colors_ret[-4] = '#2ECC71'  # Highlight System 5
    
    bars1 = ax1.bar(labels, df['Return_val'], color=colors_ret, edgecolor='black', alpha=0.85)
    ax1.set_title('各系统累计收益对比 (2019-2023)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('累计收益 (%)', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    
    # Add value labels
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (2 if yval >= 0 else -6), 
                 f"{yval:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
                 
    # Panel B: MDD
    colors_mdd = ['#8B0000' if 'Pure Rules' in s else '#F39C12' for s in systems]
    colors_mdd[-4] = '#27AE60'  # Highlight System 5
    
    bars2 = ax2.bar(labels, df['MDD_val'], color=colors_mdd, edgecolor='black', alpha=0.85)
    ax2.set_title('各系统最大回撤对比 (2019-2023)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('最大回撤 (%)', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval - 4, 
                 f"{yval:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
                 
    plt.suptitle('图 1. 六阶段消融研究结果（累计收益与最大回撤）', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_path = os.path.join(plots_out, 'fig1_ablation_bar_REAL.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" [OK] 图 1 绘制成功 -> {out_path}")

# -------------------------------------------------------------
# 图 2. 六阶段消融研究累计净值曲线对比（117只随机A股资产）
# -------------------------------------------------------------
def plot_figure_2():
    if not os.path.exists(equity_csv):
        print("  跳过图2：equity_curves_117_pool.csv 不存在")
        return
        
    df = pd.read_csv(equity_csv, encoding='utf-8')
    df['Date'] = pd.to_datetime(df['Date'])
    
    styles = {
        'System 0-A': {'color': '#8B0000', 'ls': '--', 'lw': 1.5, 'label': 'System 0-A\n(纯规则 S4)'},
        'System 1':        {'color': '#FF6B35', 'ls': '-', 'lw': 1.5, 'label': 'System 1\n(纯ML)'},
        'System 4': {'color': '#FFA500', 'ls': '-', 'lw': 1.5, 'label': 'System 4\n(ML+ATR+BBL+TOCE)'},
        'System 5':    {'color': '#2196F3', 'ls': '-', 'lw': 2.5, 'label': 'System 5\n(完整ARMS)'},
        'MA Crossover':      {'color': '#4CAF50', 'ls': '-', 'lw': 2.5, 'label': 'MA 均线交叉\n(5/20)'},
        'B&H':                     {'color': '#9E9E9E', 'ls': ':', 'lw': 1.5, 'label': 'B&H'},
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Calculate Sharpe/MDD to annotate
    sys1_max = df['System 1'].cummax()
    sys1_dd = (df['System 1'] - sys1_max) / sys1_max
    sys1_mdd = sys1_dd.min()
    
    sys5_max = df['System 5'].cummax()
    sys5_dd = (df['System 5'] - sys5_max) / sys5_max
    sys5_mdd = sys5_dd.min()
    
    for col in df.columns:
        if col == 'Date': continue
        props = styles.get(col, {'color': 'black', 'ls': '-', 'lw': 1, 'label': col})
        curve = df[col] / df[col].iloc[0]
        ax.plot(df['Date'], curve, label=props['label'], color=props['color'], 
                linestyle=props['ls'], linewidth=props['lw'], alpha=0.85)
                
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('组合净值', fontsize=12)
    ax.set_title('图 2. 六阶段消融研究累计净值曲线对比（2019—2023，117 只随机 A 股资产）', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9, edgecolor='gray', fancybox=False)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    mdd_text = f'ML回撤压缩:\n{sys1_mdd*100:.2f}% → {sys5_mdd*100:.2f}%'
    ax.annotate(mdd_text, 
                xy=(df['Date'].iloc[int(len(df)*0.75)], 0.4), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange', alpha=0.9),
                ha='center')
                
    plt.tight_layout()
    out_path = os.path.join(plots_out, 'fig2_equity_curves_117_REAL.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f" [OK] 图 2 绘制成功 -> {out_path}")

# -------------------------------------------------------------
# 图 3. 全周期 Calmar 比率排名对比（水平条形图）
# -------------------------------------------------------------
def plot_figure_3():
    if not os.path.exists(ablation_csv):
        print("  跳过图3：sci_ablation_results_v2.csv 不存在")
        return
        
    df = pd.read_csv(ablation_csv, encoding='utf-8')
    df['Calmar_val'] = df['Calmar'].astype(float)
    
    # 按 Calmar 比率升序排列（画图时最上面是最高的）
    df_sorted = df.sort_values(by='Calmar_val', ascending=True)
    
    systems = df_sorted['System'].tolist()
    labels = [s.split(':')[0] if ':' in s else s for s in systems]
    calmars = df_sorted['Calmar_val'].tolist()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    colors = ['#16A085' if val >= 0 else '#C0392B' for val in calmars]
    # Highlight System 5
    for idx, sys in enumerate(systems):
        if 'System 5' in sys:
            colors[idx] = '#2E4053'
            
    bars = ax.barh(labels, calmars, color=colors, edgecolor='black', height=0.6, alpha=0.85)
    
    ax.axvline(x=0.0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('图 3. 各系统全周期 Calmar 比率对比与学术排名 (2019-2023)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Calmar 比率 (累计收益 / 最大回撤)', fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar in bars:
        wval = bar.get_width()
        ax.text(wval + (0.01 if wval >= 0 else -0.04), bar.get_y() + bar.get_height()/2, 
                 f"{wval:.4f}", ha='left' if wval >= 0 else 'right', va='center', fontsize=9, fontweight='bold')
                 
    plt.tight_layout()
    out_path = os.path.join(plots_out, 'fig3_calmar_ranking_REAL.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" [OK] 图 3 绘制成功 -> {out_path}")

# -------------------------------------------------------------
# 图 4. System 5 与 MA 均线交叉逐年收益对比
# -------------------------------------------------------------
def plot_figure_4():
    if not os.path.exists(yearly_csv):
        print("  跳过图4：yearly_breakdown.csv 不存在")
        return
        
    df = pd.read_csv(yearly_csv, encoding='utf-8')
    df.rename(columns={df.columns[0]: 'Year'}, inplace=True)
    
    # 数据转换百分比
    df['System 5'] = df['System 5'] * 100
    df['MA Crossover'] = df['MA Crossover'] * 100
    
    years = df['Year'].astype(str).tolist()
    
    x = np.arange(len(years))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    
    rects1 = ax.bar(x - width/2, df['System 5'], width, label='System 5: Full ARMS', color='#2196F3', edgecolor='black', alpha=0.85)
    rects2 = ax.bar(x + width/2, df['MA Crossover'], width, label='Benchmark: MA Crossover', color='#4CAF50', edgecolor='black', alpha=0.85)
    
    ax.axhline(y=0.0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('图 4. System 5 与 MA 均线交叉逐年收益率对比 (2019-2023)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel('年度收益率 (%)', fontsize=10)
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:+.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    out_path = os.path.join(plots_out, 'fig4_yearly_breakdown_REAL.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" [OK] 图 4 绘制成功 -> {out_path}")

# -------------------------------------------------------------
# 图 5. 生存偏差量化：多资产池对照对比柱状图
# -------------------------------------------------------------
def plot_figure_5():
    # 整合 49只龙头、117只随机、118只稳健池 的数据
    pools = ['49只精选龙头\n(生存偏差严重)', '117只随机池\n(原始消融)', '118只独立稳健池\n(本次检验)']
    
    # 数据定义（表6数值以及我们刚跑出的稳健性指标）
    sys5_returns = [25.66, 3.55, -10.66]
    sys5_mdds = [-16.61, -16.68, -31.84]
    
    x = np.arange(len(pools))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    # Panel A: Return
    bars1 = ax1.bar(x, sys5_returns, width, color=['#F1C40F', '#3498DB', '#9B59B6'], edgecolor='black', alpha=0.85)
    ax1.axhline(y=0.0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_title('System 5 累计收益对比', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pools)
    ax1.set_ylabel('累计收益 (%)', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (1.5 if yval >= 0 else -3.5), 
                 f"{yval:+.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
                 
    # Panel B: MDD
    bars2 = ax2.bar(x, sys5_mdds, width, color=['#E67E22', '#D35400', '#C0392B'], edgecolor='black', alpha=0.85)
    ax2.set_title('System 5 最大回撤对比', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pools)
    ax2.set_ylabel('最大回撤 (%)', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval - 2.0, 
                 f"{yval:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
                 
    plt.suptitle('图 5. 生存偏差量化：跨资产池（精选龙头 vs 随机池 vs 独立稳健个股）对照对比', 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_path = os.path.join(plots_out, 'fig5_survivorship_bias_REAL.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" [OK] 图 5 绘制成功 -> {out_path}")

if __name__ == '__main__':
    print("=" * 60)
    print(" 正在一键绘制论文中的所有真实数据图表...")
    print("=" * 60)
    
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    
    print("\n所有真实图表已生成于：", plots_out)
    print("=" * 60)
