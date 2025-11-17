import pandas as pd

# Load the CSV file
df = pd.read_csv("knowledge/predictions.csv")  # Replace with your actual file path
# df = pd.read_csv("runs_artifact/knowledge_06_04:22:40_m_1/predictions.csv")  # Replace with your actual file path
# df = pd.read_csv("runs_artifact/knowledge_20250903_190352/predictions.csv")  # Replace with your actual file path


# Group by 'model_used' and calculate mean of other numeric columns
avg_stats = df.groupby("model_used")[["confidence", "inference_time", "energy_uJ"]].mean()

# Optional: round values for readability
avg_stats = avg_stats.round(4)

# Print the result
print("Average statistics per model:")
print(avg_stats)
