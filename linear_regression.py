import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. Load the dataset
# ==========================================

# Load the training and testing datasets
df_train = pd.read_csv("/content/sample_data/california_housing_train.csv")
df_test = pd.read_csv("/content/sample_data/california_housing_test.csv")

# Concatenate them into a single DataFrame
df = pd.concat([df_train, df_test], ignore_index=True)

print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

# ==========================================
# 2. Check for missing values
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values in total_bedrooms
df["total_bedrooms"] = df["total_bedrooms"].fillna(
    df["total_bedrooms"].median()
)

# ==========================================
# 3. Convert categorical column to numbers
# ==========================================

# The california housing dataset does not have 'ocean_proximity' or any categorical columns.
# This step will be skipped or adjusted if a suitable categorical column is identified.
# For now, if 'ocean_proximity' is not present, this block can be removed or commented out.
# If it's expected, please confirm the dataset.
# Based on the provided column names, 'ocean_proximity' is not in the California Housing dataset.
# Therefore, this step is not needed for this dataset.
# df = pd.get_dummies(
#     df,
#     columns=["ocean_proximity"],
#     drop_first=True
# )

print("\nColumns after encoding:")
print(df.columns)

# ==========================================
# 4. Separate X and y
# ==========================================

X = df.drop("median_house_value", axis=1)

y = df["median_house_value"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# ==========================================
# 5. Train-Test Split
# 80% training / 20% testing
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# ==========================================
# 6. Linear Regression
# ==========================================

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

# ==========================================
# 7. Evaluate Linear Regression
# ==========================================

rmse_lr = np.sqrt(
    mean_squared_error(y_test, y_pred_lr)
)

mae_lr = mean_absolute_error(
    y_test,
    y_pred_lr
)

r2_lr = r2_score(
    y_test,
    y_pred_lr
)

print("\n========== Linear Regression ==========")
print("RMSE:", rmse_lr)
print("MAE :", mae_lr)
print("R2  :", r2_lr)

# ==========================================
# 8. Coefficients
# ==========================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
})

print("\n========== Coefficients ==========")
print(coefficients)

# ==========================================
# 9. Standard Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nScaled training data:", X_train_scaled.shape)
print("Scaled testing data:", X_test_scaled.shape)
