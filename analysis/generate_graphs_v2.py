import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('results_all.csv')
sns.set_theme(style="whitegrid")

# Final evaluation score by configuration.
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='config', y='evaluation_score_final', palette='cool')
plt.title('Final Evaluation Score by Model Configuration')
plt.ylabel('Evaluation Score')
plt.savefig('graph9_evaluation_score.png', bbox_inches='tight')
plt.close()

# Computation time against path length.
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='path_length', y='computation_time_ms', hue='config', style='config', s=100, palette='deep')
plt.title('Computation Time vs. Path Length')
plt.xlabel('Path Length (Steps)')
plt.ylabel('Computation Time (ms)')
plt.legend(title='Configuration')
plt.savefig('graph10_time_vs_length.png', bbox_inches='tight')
plt.close()

print("Saved graph9_evaluation_score.png and graph10_time_vs_length.png.")