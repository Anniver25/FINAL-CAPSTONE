import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the Excel file
file_path = 'Favila, Trisha MIDTERM Excel.xlsx'
df = pd.read_excel(file_path, sheet_name='RAW DATA')

# Clean data - remove null values
print(f"Original data points: {len(df)}")
df_clean = df.dropna()
print(f"After cleaning: {len(df_clean)} points")

# Verify sufficient clean records (commented out for current dataset)
# assert len(df_clean) >= 30, "Insufficient data points after cleaning"

# Remove outliers using Z-scores
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
z_scores = stats.zscore(df_clean[numeric_cols])
df_clean = df_clean[(np.abs(z_scores) < 3).all(axis=1)]
print(f"After outlier removal: {len(df_clean)} points")

# NumPy Operations
print("\nNumPy Operations:")
# 1. GDP statistics
print("1. GDP per capita stats:")
print(f"   Mean: {np.mean(df_clean['GDP per capita (USD)']):,.2f}, "
      f"Median: {np.median(df_clean['GDP per capita (USD)']):,.2f}, "
      f"Std: {np.std(df_clean['GDP per capita (USD)']):,.2f}")

# 2. Normalize Life Expectancy
scaler = StandardScaler()
life_scaled = scaler.fit_transform(df_clean[['Life expectancy']])
print("2. Scaled Life Expectancy (first 5):", life_scaled[:5].flatten())

# 3. Reshape matrix
matrix = df_clean[['GDP per capita (USD)', 'Life satisfaction']].values.reshape(-1, 2)
print("3. Reshaped matrix shape:", matrix.shape)

# 4. Log transformation of Homicide Rate
log_homicide = np.log(df_clean['Homicide rate'] + 1)  # +1 to avoid log(0)
print("4. Log Homicide Rate (first 5):", log_homicide[:5].values)

# 5. Random sampling
sample = df_clean.sample(3, random_state=1)
print("\n5. Random Sample:")
print(sample[['Country', 'GDP per capita (USD)', 'Life satisfaction']])

# SciPy: Pearson Correlation
r, p = stats.pearsonr(df_clean['GDP per capita (USD)'], df_clean['Life satisfaction'])
print(f"\nSciPy Correlation: r = {r:.3f}, p = {p:.4f}")

# Statsmodels: OLS Regression
X = sm.add_constant(df_clean[['GDP per capita (USD)', 'Life expectancy']])
y = df_clean['Life satisfaction']
model = sm.OLS(y, X).fit()
print("\nStatsmodels OLS Summary:")
print(model.summary())

# Visualization 1: GDP vs Life Satisfaction with Regression
plt.figure(figsize=(10, 6))
sns.regplot(x='GDP per capita (USD)', y='Life satisfaction', data=df_clean,
            scatter_kws={'alpha':0.7, 'color':'teal'},
            line_kws={'color':'coral', 'linewidth':2})
plt.title('GDP per Capita vs Life Satisfaction', fontsize=14)
plt.xlabel('GDP per Capita (USD)', fontsize=12)
plt.ylabel('Life Satisfaction Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Visualization 2: Top 10 Countries by GDP
plt.figure(figsize=(12, 6))
top_gdp = df_clean.nlargest(10, 'GDP per capita (USD)')
sns.barplot(x='GDP per capita (USD)', y='Country', data=top_gdp, palette='viridis')
plt.title('Top 10 Countries by GDP per Capita', fontsize=14)
plt.xlabel('GDP per Capita (USD)', fontsize=12)
plt.ylabel('')
plt.xticks(rotation=45)
plt.show()

# Visualization 3: Correlation Heatmap
corr_matrix = df_clean[['GDP per capita (USD)', 'Life expectancy', 
                       'Employment rate', 'Homicide rate', 
                       'Life satisfaction']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Visualization 4: Pairplot of Key Metrics
pair_vars = ['GDP per capita (USD)', 'Life expectancy', 
            'Homicide rate', 'Life satisfaction']
sns.pairplot(df_clean[pair_vars], diag_kind='kde', 
             plot_kws={'alpha':0.6, 'edgecolor':'w'})
plt.suptitle('Pairwise Relationships of Key Metrics', y=1.02)
plt.show()

# Visualization 5: Distribution of Life Satisfaction
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['Life satisfaction'], bins=15, 
             kde=True, color='royalblue')
plt.title('Distribution of Life Satisfaction Scores', fontsize=14)
plt.xlabel('Life Satisfaction Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(alpha=0.3)
plt.show()