import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('s.csv')

# Clean data - remove null values
print(f"Original data points: {len(df)}")
df_clean = df.dropna()
print(f"After cleaning: {len(df_clean)} points")

# Ensure there are enough records
assert len(df_clean) >= 100, "Dataset must have at least 100 points after cleaning"

# Optional: Remove outliers using z-score
z_scores = stats.zscore(df_clean.select_dtypes(include=[np.number]))
df_clean = df_clean[(np.abs(z_scores) < 3).all(axis=1)]
print(f"After outlier removal: {len(df_clean)} points")

# NumPy Operations
print("\nNumPy Operations:")
# 1. Mean, Median, Std of math score
print("1. Math Score stats:")
print(f"   Mean: {np.mean(df_clean['math score']):.2f}, Median: {np.median(df_clean['math score']):.2f}, Std: {np.std(df_clean['math score']):.2f}")

# 2. Normalize reading score
scaler = StandardScaler()
read_scaled = scaler.fit_transform(df_clean[['reading score']])
print("2. Scaled Reading Scores (first 5):", read_scaled[:5].flatten())

# 3. Reshape a matrix
matrix = df_clean[['math score', 'reading score']].values.reshape(-1, 2)
print("3. Reshaped matrix shape:", matrix.shape)

# 4. Log transformation of writing score (+1 to avoid log(0))
log_write = np.log(df_clean['writing score'] + 1)
print("4. Log Writing Score (first 5):", log_write[:5].values)

# 5. Random sampling
sample = df_clean.sample(5, random_state=42)
print("5. Random Sample:")
print(sample[['gender', 'math score', 'reading score', 'writing score']])

# SciPy: Pearson correlation between reading and writing scores
r, p = stats.pearsonr(df_clean['reading score'], df_clean['writing score'])
print(f"\nSciPy Correlation (Reading vs Writing): r = {r:.3f}, p = {p:.4f}")

# Statsmodels: OLS Regression - Predict writing score using reading score
X = sm.add_constant(df_clean['reading score'])
y = df_clean['writing score']
model = sm.OLS(y, X).fit()
print("\nStatsmodels OLS Summary:")
print(model.summary())

# Visualization 1: Histogram of Math Score
plt.figure(figsize=(8,6))
sns.histplot(df_clean['math score'], kde=True, bins=20)
plt.title('Distribution of Math Scores')
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.show()

# Visualization 2: Boxplot of Scores by Gender
plt.figure(figsize=(10,6))
sns.boxplot(x='gender', y='math score', data=df_clean)
plt.title('Math Score by Gender')
plt.show()

# Bonus Visualization: Pairplot of scores
sns.pairplot(df_clean[['math score', 'reading score', 'writing score']])
plt.suptitle('Pairwise Relationships between Scores', y=1.02)
plt.show()

# Correlation plot: Reading vs Writing Score
plt.figure(figsize=(8,6))
sns.regplot(x='reading score', y='writing score', data=df_clean,
            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Reading Score vs Writing Score')
plt.xlabel('Reading Score')
plt.ylabel('Writing Score')
plt.show()
