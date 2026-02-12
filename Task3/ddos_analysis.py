# ddos_analysis.py - Task 3 revised version
# Adjusted to detect spikes in your 20-minute log

import pandas as pd
import re
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


def parse_log(file_path):
    timestamps = []
    timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\]'

    line_count = 0
    parsed_count = 0

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue

            match = re.search(timestamp_pattern, line)
            if match:
                ts_str = match.group(1)
                try:
                    ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S%z')
                    timestamps.append(ts)
                    parsed_count += 1
                except ValueError:
                    pass

    print(f"Total lines read: {line_count}")
    print(f"Successfully parsed timestamps: {parsed_count}")

    if parsed_count == 0:
        raise ValueError("No timestamps parsed.")

    df = pd.DataFrame({'timestamp': timestamps})
    df['minute'] = df['timestamp'].dt.floor('min')
    requests_per_min = df.groupby('minute').size().reset_index(name='count')
    requests_per_min['minutes_since_start'] = (
            (requests_per_min['minute'] - requests_per_min['minute'].min())
            .dt.total_seconds() / 60
    )
    return requests_per_min


if __name__ == "__main__":
    print("=== Task 3 - DDoS Detection (Revised) ===\n")

    # Parse
    df = parse_log('server.log')
    print(f"Time range: {df['minute'].min()} → {df['minute'].max()}")
    print(f"Total minutes: {len(df)}\n")
    print("First 5 minutes:\n", df.head(), "\n")

    # ─── Linear Regression ────────────────────────────────────────
    X = sm.add_constant(df['minutes_since_start'])
    y = df['count']
    model = sm.OLS(y, X).fit()
    df['predicted'] = model.predict(X)
    df['residual'] = df['count'] - df['predicted']

    print("Linear Regression summary:")
    print(f"Intercept: {model.params['const']:.2f}")
    print(f"Slope: {model.params['minutes_since_start']:.2f}")
    print(f"R²: {model.rsquared:.4f}\n")

    # ─── Anomaly detection ────────────────────────────────────────
    # Lower threshold to catch spikes
    threshold_linear = df['residual'].mean() + 1.8 * df['residual'].std()
    attacks_linear = df[df['residual'] > threshold_linear].copy()

    print(f"Linear residual threshold (1.8σ): {threshold_linear:.2f}")
    print("Detected spikes (linear method):")
    if attacks_linear.empty:
        print("None")
    else:
        print(attacks_linear[['minute', 'count', 'residual']].to_string(index=False))

    # ─── Rolling mean + std (better for bursts) ───────────────────
    window = 5
    df['rolling_mean'] = df['count'].rolling(window=window, min_periods=1, center=True).mean()
    df['rolling_std'] = df['count'].rolling(window=window, min_periods=1, center=True).std()
    df['rolling_threshold'] = df['rolling_mean'] + 3 * df['rolling_std']

    attacks_rolling = df[df['count'] > df['rolling_threshold']].copy()

    print(f"\nRolling threshold (3σ):")
    print("Detected spikes (rolling method):")
    if attacks_rolling.empty:
        print("None")
    else:
        print(attacks_rolling[['minute', 'count', 'rolling_mean', 'rolling_threshold']].to_string(index=False))

    # ─── Plot 1: Linear Regression ────────────────────────────────
    plt.figure(figsize=(14, 6))
    plt.plot(df['minutes_since_start'], df['count'], label='Actual', color='blue', alpha=0.7)
    plt.plot(df['minutes_since_start'], df['predicted'], label='Linear trend', color='red', lw=2, ls='--')
    plt.axhline(df['predicted'].mean() + threshold_linear, color='orange', ls=':', label='1.8σ threshold')

    if not attacks_linear.empty:
        plt.scatter(attacks_linear['minutes_since_start'], attacks_linear['count'],
                    color='red', s=120, label='Detected (linear)', zorder=10)

    plt.title('DDoS Detection - Linear Regression')
    plt.xlabel('Minutes since start')
    plt.ylabel('Requests per minute')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ddos_linear_plot.png', dpi=150)
    plt.close()

    # ─── Plot 2: Rolling method ───────────────────────────────────
    plt.figure(figsize=(14, 6))
    plt.plot(df['minutes_since_start'], df['count'], label='Actual', color='blue', alpha=0.7)
    plt.plot(df['minutes_since_start'], df['rolling_mean'], label=f'Rolling mean ({window} min)', color='green', lw=2)
    plt.fill_between(df['minutes_since_start'],
                     df['rolling_mean'] - 3 * df['rolling_std'],
                     df['rolling_mean'] + 3 * df['rolling_std'],
                     color='orange', alpha=0.15, label='±3σ band')

    if not attacks_rolling.empty:
        plt.scatter(attacks_rolling['minutes_since_start'], attacks_rolling['count'],
                    color='red', s=120, label='Detected (rolling)', zorder=10)

    plt.title('DDoS Detection - Rolling Mean & Std')
    plt.xlabel('Minutes since start')
    plt.ylabel('Requests per minute')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ddos_rolling_plot.png', dpi=150)
    plt.close()

    print("\nPlots saved:")
    print("• ddos_linear_plot.png")
    print("• ddos_rolling_plot.png")
    print("\nDone.")