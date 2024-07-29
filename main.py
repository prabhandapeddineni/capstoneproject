import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('file:///Users/prabha/Downloads/E-commerce%20Customer%20Behavior%20-%20Sheet1.csv')

# Print the column names to verify
print("Columns in the dataset:", data.columns)

print(f"\nNumber of Nulls: {data.isnull().sum()}")
# Data cleaning
data.drop_duplicates(inplace=True)
data.ffill(inplace=True)  # Fill missing values forward
print(f"\nNumber of Nulls: {data.isnull().sum()}")

# Membership distribution
data['Membership Type'].value_counts().plot(kind='bar', title='Membership Type Distribution')
plt.xlabel('Membership Type')
plt.ylabel('Count')
plt.show()

# Spending patterns
data.groupby('Membership Type')['Total Spend'].mean().plot(kind='bar', title='Average Total Spend by Membership Type')
plt.xlabel('Membership Type')
plt.ylabel('Average Total Spend')
plt.show()

# Bar plot for purchase frequency (Days Since Last Purchase) by membership type
data.groupby('Membership Type')['Days Since Last Purchase'].mean().plot(kind='bar', title='Average Days Since Last Purchase by Membership Type')
plt.xlabel('Membership Type')
plt.ylabel('Average Days Since Last Purchase')
plt.show()

# Bar plot for average items purchased by membership type
data.groupby('Membership Type')['Items Purchased'].mean().plot(kind='bar', title='Average Items Purchased by Membership Type')
plt.xlabel('Membership Type')
plt.ylabel('Average Items Purchased')
plt.show()

# Bar plot for the total number of items purchased by membership type
data.groupby('Membership Type')['Items Purchased'].sum().plot(kind='bar', title='Total Items Purchased by Membership Type')
plt.xlabel('Membership Type')
plt.ylabel('Total Items Purchased')
plt.show()

# Satisfaction Levels
data['Satisfaction Level'].value_counts().plot(kind='bar', title='Satisfaction Level Distribution')
plt.xlabel('Satisfaction Level')
plt.ylabel('membership type')
plt.show()


# Encode the satisfaction levels
label_encoder = LabelEncoder()
data['Satisfaction Level'] = label_encoder.fit_transform(data['Satisfaction Level'])

# Predictive modeling
# Define the features and target variable
X = data[['Total Spend', 'Items Purchased']]
y = data['Satisfaction Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Model coefficients
print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')


# Scatter plot of actual vs predicted satisfaction levels
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Satisfaction Level')
plt.ylabel('Predicted Satisfaction Level')
plt.title('Actual vs Predicted Satisfaction Levels')
plt.show()


# clustering analyisis:
# K-Means Clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(data[['Total Spend', 'Satisfaction Level']])

# Plot clusters
plt.scatter(data['Total Spend'], data['Satisfaction Level'], c=data['Cluster'])
plt.title('Customer Segments')
plt.xlabel('Total Spend')
plt.ylabel('Satisfaction Level')
plt.show()

