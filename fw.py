import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('g.csv')

# Clean data - remove null values
print(f"Original data points: {len(df)}")
df_clean = df.dropna()
print(f"After cleaning: {len(df_clean)} points")

# Verify at least 100 clean records
assert len(df_clean) >= 100, "Dataset must have at least 100 points after cleaning"

# Optional bonus: Remove outliers
z_scores = stats.zscore(df_clean.select_dtypes(include=[np.number]))
df_clean = df_clean[(np.abs(z_scores) < 3).all(axis=1)]
print(f"After outlier removal: {len(df_clean)} points")

# NumPy Operations
print("\nNumPy Operations:")
# 1. Mean, Median, Std of Total Waste
print("1. Total Waste stats:")
print(f"   Mean: {np.mean(df_clean['Total Waste (Tons)']):.2f}, Median: {np.median(df_clean['Total Waste (Tons)']):.2f}, Std: {np.std(df_clean['Total Waste (Tons)']):.2f}")

# 2. Normalize Economic Loss
scaler = StandardScaler()
econ_scaled = scaler.fit_transform(df_clean[['Economic Loss (Million $)']])
print("2. Scaled Economic Loss (first 5):", econ_scaled[:5].flatten())

# 3. Reshape a matrix
matrix = df_clean[['Total Waste (Tons)', 'Population (Million)']].values.reshape(-1, 2)
print("3. Reshaped matrix shape:", matrix.shape)

# 4. Log transformation
log_waste = np.log(df_clean['Total Waste (Tons)'])
print("4. Log Waste (first 5):", log_waste[:5].values)

# 5. Random sampling
sample = df_clean.sample(5, random_state=1)
print("5. Random Sample:")
print(sample[['Country', 'Total Waste (Tons)', 'Economic Loss (Million $)']])

# SciPy Operation: Pearson Correlation
r, p = stats.pearsonr(df_clean['Avg Waste per Capita (Kg)'], df_clean['Economic Loss (Million $)'])
print(f"\nSciPy Correlation: r = {r:.3f}, p = {p:.4f}")

# Statsmodels: OLS Regression
X = sm.add_constant(df_clean['Avg Waste per Capita (Kg)'])
y = df_clean['Economic Loss (Million $)']
model = sm.OLS(y, X).fit()
print("\nStatsmodels OLS Summary:")
print(model.summary())

# Visualization 1: Histogram of Waste per Capita
plt.figure(figsize=(8,6))
sns.histplot(df_clean['Avg Waste per Capita (Kg)'], kde=True, bins=20)
plt.title('Distribution of Waste per Capita')
plt.xlabel('Kg per Capita')
plt.ylabel('Frequency')
plt.show()

# Visualization 2: Boxplot by Food Category
plt.figure(figsize=(12,6))
sns.boxplot(x='Food Category', y='Total Waste (Tons)', data=df_clean)
plt.title('Waste by Food Category')
plt.xticks(rotation=45)
plt.show()

# Bonus Visualization: Pairplot
sns.pairplot(df_clean[['Total Waste (Tons)', 'Economic Loss (Million $)', 'Avg Waste per Capita (Kg)', 'Population (Million)']])
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()

# Correlation plot with regression line
plt.figure(figsize=(8,6))
sns.regplot(x='Total Waste (Tons)', y='Economic Loss (Million $)', data=df_clean,
            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Total Waste vs Economic Loss')
plt.xlabel('Total Waste (Tons)')
plt.ylabel('Economic Loss (Million $)')
plt.show()



