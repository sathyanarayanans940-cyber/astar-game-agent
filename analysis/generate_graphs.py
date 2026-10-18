import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the benchmark results.
df = pd.read_csv('results_all.csv')

# Use the same theme across plots.
sns.set_theme(style="whitegrid")

# 1. Success Rate Comparison
plt.figure(figsize=(8, 5))
success_rates = df.groupby('config')['success'].mean() * 100
sns.barplot(x=success_rates.index, y=success_rates.values, palette='viridis')
plt.title('Success Rate by Model Configuration')
plt.ylabel('Success Rate (%)')
plt.savefig('graph1_success_rate.png')
plt.close()

# 2. Average Computation Time per Config
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='config', y='computation_time_ms', palette='Set2')
plt.title('Computation Time (ms) by Configuration')
plt.ylabel('Computation Time (ms)')
plt.savefig('graph2_comp_time.png')
plt.close()

# 3. Nodes Expanded Comparison
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='config', y='nodes_expanded', errorbar=None, palette='mako')
plt.title('Average Nodes Expanded per Model')
plt.ylabel('Nodes Expanded')
plt.savefig('graph3_nodes.png')
plt.close()

# 4. Path Efficiency (Actual vs Optimal)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='optimal_path_cost', y='actual_path_cost', hue='config', style='config', s=100)
plt.plot([0, df['optimal_path_cost'].max()], [0, df['optimal_path_cost'].max()], 'r--', label='Ideal Efficiency')
plt.title('Actual Path Cost vs Optimal Path Cost')
plt.legend()
plt.savefig('graph4_path_efficiency.png')
plt.close()

# 5. Total Steps Taken across Episodes
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='episode', y='total_steps', hue='config', marker='o')
plt.title('Total Steps Taken per Episode')
plt.ylabel('Total Steps')
plt.savefig('graph5_total_steps.png')
plt.close()

# 6. Enemy Encounters / Near Collisions
plt.figure(figsize=(8, 5))
encounters = df.groupby('config')[['enemy_encounters_count', 'near_collision_count']].mean()
encounters.plot(kind='bar', figsize=(8,5), colormap='coolwarm')
plt.title('Average Enemy Encounters & Near Collisions')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('graph6_collisions.png')
plt.close()

# 7. Replans Count by Model
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='config', y='replans_count', errorbar=None, palette='rocket')
plt.title('Average Replans Triggered During Navigation')
plt.ylabel('Number of Replans')
plt.savefig('graph7_replans.png')
plt.close()

# 8. Average Distance Maintained from Enemies
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='config', y='avg_enemy_distance', palette='magma')
plt.title('Average Distance Maintained from Enemies')
plt.ylabel('Distance (Grid Cells)')
plt.savefig('graph8_enemy_distance.png')
plt.close()

print("Graphs successfully generated and saved as PNG files!")