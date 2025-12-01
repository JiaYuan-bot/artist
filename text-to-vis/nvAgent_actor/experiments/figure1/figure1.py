import matplotlib.pyplot as plt
import numpy as np

# Set up the figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Define data for each dataset
datasets = [{
    'name': 'Text-to-SQL',
    'artist_latency': 38.49,
    'artist_pr': 0.6781,
    'default_latency': 224.30,
    'default_pr': 0.6956,
    'cascade_0.2_latency': 99.82,
    'cascade_0.2_pr': 0.6351,
    'cascade_0.5_latency': 113.42,
    'cascade_0.5_pr': 0.6604,
    'cascade_0.8_latency': 155.13,
    'cascade_0.8_pr': 0.6890,
    'LLM_planning_latency': 83.04,
    'LLM_planning_pr': 0.6217,
    'cognify_latencys': [100, 200],
    'cognify_prs': [0.65, 0.67],
    'speedup': '2.41X faster',
    'improvement': '16% higher'
}, {
    'name': 'Text-to-Visualization',
    'artist_latency': 6.92,
    'artist_pr': 0.6667,
    'default_latency': 36.01,
    'default_pr': 0.6725,
    'cascade_0.2_latency': 11.64,
    'cascade_0.2_pr': 0.6020,
    'cascade_0.5_latency': 31.12,
    'cascade_0.5_pr': 0.6395,
    'cascade_0.8_latency': 38.38,
    'cascade_0.8_pr': 0.6432,
    'LLM_planning_latency': 15.01,
    'LLM_planning_pr': 0.6279,
    'cognify_latencys': [20, 30],
    'cognify_prs': [0.61, 0.62],
    'speedup': '2.41X faster',
    'improvement': '16% higher'
}, {
    'name': 'Advanced QA',
    'artist_latency': 29.16,
    'artist_pr': 0.488,
    'default_latency': 311.58,
    'default_pr': 0.505,
    'cascade_0.2_latency': 19.32,
    'cascade_0.2_pr': 0.433,
    'cascade_0.5_latency': 19.32,
    'cascade_0.5_pr': 0.443,
    'cascade_0.8_latency': 19.32,
    'cascade_0.8_pr': 0.443,
    'LLM_planning_latency': 96.91,
    'LLM_planning_pr': 0.416,
    'cognify_latencys': [20, 30],
    'cognify_prs': [0.41, 0.42],
    'speedup': '2.41X faster',
    'improvement': '16% higher'
}]

# Plot each dataset
for idx, (ax, data) in enumerate(zip(axes, datasets)):
    if idx == 0 or idx == 1:
        # Plot ARTIST (green circle)
        ax.scatter(data['artist_latency'],
                   data['artist_pr'],
                   color='green',
                   s=100,
                   marker='o',
                   zorder=5,
                   label='ARTIST')

        # Plot Default (orange diamond)
        ax.scatter(data['default_latency'],
                   data['default_pr'],
                   color='orange',
                   s=100,
                   marker='x',
                   zorder=4,
                   label='Default')

        # Plot Cascade methods (purple triangles)
        ax.scatter(data['cascade_0.2_latency'],
                   data['cascade_0.2_pr'],
                   color='purple',
                   s=100,
                   marker='^',
                   zorder=4,
                   label='Cascade 0.2')
        ax.scatter(data['cascade_0.5_latency'],
                   data['cascade_0.5_pr'],
                   color='purple',
                   s=100,
                   marker='^',
                   zorder=4)
        ax.scatter(data['cascade_0.8_latency'],
                   data['cascade_0.8_pr'],
                   color='purple',
                   s=100,
                   marker='^',
                   zorder=4)

        # Plot LLM Planning (brown square)
        ax.scatter(data['LLM_planning_latency'],
                   data['LLM_planning_pr'],
                   color='brown',
                   s=100,
                   marker='s',
                   zorder=4,
                   label='LLM Planning')

        # Plot Cognify (blue X marks)
        # ax.scatter(data['cognify_latencys'],
        #            data['cognify_prs'],
        #            color='blue',
        #            s=100,
        #            marker='D',
        #            linewidths=2,
        #            zorder=3,
        #            label='Cognify')
    else:
        # Plot ARTIST (green circle)
        ax.scatter(data['artist_latency'],
                   data['artist_pr'],
                   color='green',
                   s=100,
                   marker='o',
                   zorder=5,
                   label='ARTIST')

        # Plot Default (orange diamond)
        ax.scatter(data['default_latency'],
                   data['default_pr'],
                   color='orange',
                   s=100,
                   marker='x',
                   zorder=4,
                   label='Default')

        # Plot LLM Planning (brown square)
        ax.scatter(data['LLM_planning_latency'],
                   data['LLM_planning_pr'],
                   color='brown',
                   s=100,
                   marker='s',
                   zorder=4,
                   label='LLM Planning')

    # # Add arrows and annotations
    # # Calculate the rightmost baseline method for speedup comparison
    # baseline_latency = data['default_latency']
    # baseline_pr = data['default_pr']

    # # Horizontal arrow for speedup
    # arrow_y = (data['artist_pr'] + baseline_pr) / 2
    # ax.annotate('',
    #             xy=(baseline_latency, arrow_y),
    #             xytext=(data['artist_latency'], arrow_y),
    #             arrowprops=dict(arrowstyle='<->', color='green', lw=2))

    # # Calculate speedup
    # speedup = baseline_latency / data['artist_latency']
    # speedup_text = f'{speedup:.2f}X faster'

    # # Speedup text
    # mid_x = (data['artist_latency'] + baseline_latency) / 2
    # ax.text(mid_x,
    #         arrow_y + 0.01,
    #         speedup_text,
    #         ha='center',
    #         va='bottom',
    #         fontsize=10,
    #         weight='bold')

    # # Vertical arrow for improvement
    # max_pr = data['default_pr']
    # arrow_x = data['artist_latency'] * 0.8
    # ax.annotate('',
    #             xy=(arrow_x, data['artist_pr']),
    #             xytext=(arrow_x, max_pr),
    #             arrowprops=dict(arrowstyle='<->', color='green', lw=2))

    # # Calculate improvement percentage
    # improvement = ((data['artist_pr'] - max_pr) / max_pr) * 100
    # improvement_text = f'{improvement:.0f}% higher' if improvement > 0 else f'{abs(improvement):.0f}% lower'

    # # Improvement text
    # mid_y = (data['artist_pr'] + max_pr) / 2
    # ax.text(arrow_x * 0.85,
    #         mid_y,
    #         improvement_text,
    #         ha='center',
    #         va='center',
    #         fontsize=10,
    #         weight='bold',
    #         rotation=90)

    # Set labels and title
    ax.set_xlabel('Average Latency (s)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(f"Task: {data['name']}", fontsize=12, weight='bold')

    # Set grid
    ax.grid(True, alpha=0.3)

# Add legend to the top
legend_labels = [
    'ARTIST', 'Default', 'Cascade (0.2, 0.5, 0.8)', 'LLM Planning', 'Cognify'
]
legend_markers = [
    plt.Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor='green',
               markersize=10),
    plt.Line2D([0], [0],
               marker='x',
               color='w',
               markeredgecolor='orange',
               markersize=10),
    plt.Line2D([0], [0],
               marker='^',
               color='w',
               markerfacecolor='purple',
               markersize=10),
    plt.Line2D([0], [0],
               marker='s',
               color='w',
               markerfacecolor='brown',
               markersize=10),
    plt.Line2D([0], [0],
               marker='D',
               color='w',
               markerfacecolor='blue',
               markersize=10,
               markeredgewidth=2)
]

fig.legend(legend_markers,
           legend_labels,
           loc='upper center',
           bbox_to_anchor=(0.5, 1.05),
           ncol=5,
           fontsize=10,
           frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
print("Figure saved to /mnt/user-data/outputs/rag_comparison.png")
plt.savefig('comparison.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.show()
