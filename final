# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, header=None, names=columns)

# Check for missing values
print(iris_df.isnull().sum())

# Standardizing the numerical features
scaler = StandardScaler()
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_df[features] = scaler.fit_transform(iris_df[features])

# Summary statistics grouped by species
summary_stats = iris_df.groupby('species').agg(['mean', 'median', 'var'])
print(summary_stats)

# Boxplot for each feature
for feature in features:
    sns.boxplot(x='species', y=feature, data=iris_df)
    plt.title(f'Boxplot of {feature} by Species')
    plt.show()

# Violin plot for petal_length by species
sns.violinplot(x='species', y='petal_length', data=iris_df)
plt.title('Violin Plot of Petal Length by Species')
plt.show()

# Scatter plot of sepal_length vs sepal_width
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris_df)
plt.title('Sepal Length vs Sepal Width by Species')
plt.show()

# Pair plot using seaborn
sns.pairplot(iris_df, hue='species')
plt.show()

# Species distribution
species_count = iris_df['species'].value_counts()
species_count.plot(kind='bar')
plt.title('Species Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# Percentage distribution of each species
species_percentage = iris_df['species'].value_counts(normalize=True) * 100
print(species_percentage)

# Correlation matrix
correlation_matrix = iris_df[features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()
