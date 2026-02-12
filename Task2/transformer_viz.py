# transformer_viz.py
# Run this script to generate the two required images for task_2 folder:
#   - attention_heatmap.png
#   - positional_encoding.png

import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
# 1. Self-Attention Heatmap Visualization (simplified)
# ────────────────────────────────────────────────
def plot_attention_heatmap():
    # Simulate a 10×10 attention matrix (in real transformers it's seq_len × seq_len)
    seq_len = 10
    attention_weights = np.random.rand(seq_len, seq_len)  # random values [0,1]
    
    # Make diagonal stronger (common in real attention)
    np.fill_diagonal(attention_weights, 0.9)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.title('Simplified Self-Attention Heatmap\n(Brighter = higher attention between tokens)')
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    plt.xticks(range(seq_len))
    plt.yticks(range(seq_len))
    plt.grid(False)
    
    plt.savefig('attention_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: attention_heatmap.png")


# ────────────────────────────────────────────────
# 2. Positional Encoding Visualization
# ────────────────────────────────────────────────
def get_positional_encoding(seq_len=50, d_model=128):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def plot_positional_encoding():
    pe = get_positional_encoding(seq_len=50, d_model=128)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(pe, cmap='viridis', aspect='auto')
    plt.title('Positional Encoding\n(Sinusoidal functions for positions 0–49)')
    plt.xlabel('Embedding dimension')
    plt.ylabel('Position in sequence')
    plt.colorbar(label='Value')
    
    plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: positional_encoding.png")


# ────────────────────────────────────────────────
# Run both visualizations
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating visualizations for Task 2...")
    plot_attention_heatmap()
    plot_positional_encoding()
    print("Done! Check the files: attention_heatmap.png and positional_encoding.png")