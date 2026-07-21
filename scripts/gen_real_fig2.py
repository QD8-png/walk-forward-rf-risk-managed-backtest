# -*- coding: utf-8 -*-
"""Generate Real Figure 2: Equity curves for 117-stock pool."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Try to use SimHei for Chinese
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

base = r'C:\Users\qwe\.gemini\antigravity\scratch\walk-forward-rf-risk-managed-backtest'
csv_path = os.path.join(base, 'equity_curves_117_pool.csv')

if not os.path.exists(csv_path):
    print("CSV not found yet!")
    sys.exit(1)

df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])

# Colors and styles
styles = {
    'System 0-A\n(纯规则 S4)': {'color': '#8B0000', 'ls': '--', 'lw': 1.5},
    'System 1\n(纯ML)':        {'color': '#FF6B35', 'ls': '-', 'lw': 1.5},
    'System 4\n(ML+ATR+BBL+TOCE)': {'color': '#FFA500', 'ls': '-', 'lw': 1.5},
    'System 5\n(完整ARMS)':    {'color': '#2196F3', 'ls': '-', 'lw': 2.5},
    'MA均线交叉\n(5/20)':      {'color': '#4CAF50', 'ls': '-', 'lw': 2.5},
    'B&H':                     {'color': '#9E9E9E', 'ls': ':', 'lw': 1.5},
}

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Calculate the actual MDD for System 1 and System 5 for the annotation
sys1_max = df['System 1\n(纯ML)'].cummax()
sys1_dd = (df['System 1\n(纯ML)'] - sys1_max) / sys1_max
sys1_mdd = sys1_dd.min()

sys5_max = df['System 5\n(完整ARMS)'].cummax()
sys5_dd = (df['System 5\n(完整ARMS)'] - sys5_max) / sys5_max
sys5_mdd = sys5_dd.min()

for col in df.columns:
    if col == 'Date': continue
    
    props = styles.get(col, {'color': 'black', 'ls': '-', 'lw': 1})
    
    # Normalize to 1 at start
    curve = df[col] / df[col].iloc[0]
    
    ax.plot(df['Date'], curve, label=col, color=props['color'], 
            linestyle=props['ls'], linewidth=props['lw'], alpha=0.85)

ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('组合净值', fontsize=12)
ax.set_title('图 2. 六阶段消融研究累计净值曲线对比（2019—2023，117 只随机 A 股资产）', 
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9, 
          edgecolor='gray', fancybox=False)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())

# Add annotation based on ACTUAL calculated data
mdd_text = f'ML回撤压缩:\n{sys1_mdd*100:.2f}% → {sys5_mdd*100:.2f}%'
ax.annotate(mdd_text, 
            xy=(df['Date'].iloc[int(len(df)*0.7)], 0.4), fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange', alpha=0.9),
            ha='center')

plt.tight_layout()

# Save
out_base = r'C:\Users\qwe\.gemini\antigravity\brain\d23e36ec-83a1-4edf-bd74-a626e97618aa\figures'
fig2_path = os.path.join(out_base, 'fig2_equity_curves_117_REAL.png')
plt.savefig(fig2_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"REAL Figure 2 saved to: {fig2_path}")
plt.close()
