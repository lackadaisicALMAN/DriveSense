"""
analytics.py  –  DriveSense AI  |  Telemetry Visualisation
===========================================================
Generates beautiful matplotlib charts from session events
for inclusion in the UI dashboard and reports.
"""

import matplotlib
matplotlib.use("Agg")  # thread-safe, non-interactive backend
import matplotlib.pyplot as plt
import os
import tempfile
from typing import Optional
from detector import FrameEvent


def generate_distance_chart(events: list[FrameEvent], safe_distance: float) -> Optional[str]:
    """
    Generate a telemetry line chart of Following Distance vs Safe Distance over time.
    Saves to a temporary file and returns its path.
    """
    valid_events = [e for e in events if e.nearest_car_dist_m is not None]
    if not valid_events:
        return None
        
    times = [e.timestamp_s for e in valid_events]
    distances = [e.nearest_car_dist_m for e in valid_events]
    
    # Matplotlib styling for dark glass theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    fig.patch.set_facecolor('#0a0a14')
    ax.set_facecolor('#10101e')
    
    # Plot distances
    ax.plot(times, distances, color='#3b5bdb', label='Following Distance', linewidth=2.5)
    
    # Plot safe distance line
    ax.axhline(y=safe_distance, color='#ef4444', linestyle='--', label='Safe Distance Buffer', linewidth=2)
    
    # Highlight unsafe regions (where following distance < safe distance)
    ax.fill_between(times, distances, safe_distance, where=[d < safe_distance for d in distances],
                    color='#ef4444', alpha=0.18, interpolate=True, label='Unsafe Zone')
    
    # Add title and labels
    ax.set_title("Following Distance Telemetry Profile", fontsize=13, fontweight='bold', pad=15, color='#e8e8ff')
    ax.set_xlabel("Time (seconds)", fontsize=10, labelpad=8, color='#aaa')
    ax.set_ylabel("Distance (metres)", fontsize=10, labelpad=8, color='#aaa')
    
    # Grid customization
    ax.grid(True, color='#1e1e35', linestyle=':', linewidth=0.8)
    
    # Legend
    legend = ax.legend(loc='upper right', frameon=True, facecolor='#10101e', edgecolor='#1e1e35')
    for text in legend.get_texts():
        text.set_color('#ccc')
        
    ax.tick_params(colors='#888', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#1e1e35')
    ax.spines['bottom'].set_color('#1e1e35')
    
    # Save to temp file
    chart_path = os.path.join(tempfile.gettempdir(), f"drivesense_telemetry_chart.png")
    fig.tight_layout()
    fig.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return chart_path
