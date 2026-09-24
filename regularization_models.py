import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. Load dataset
# The dataset 'df' is already loaded and preprocessed in the previous cell.
# ==========================================

# Fill missing values
df["total_bedrooms"] = df["total_bedrooms"].fillna(
    df["total_bedrooms"].median()
)

# ==========================================
# 2. Create features like the screenshot
# ==========================================

df["AveRooms"] = df["total_rooms"] / df["households"]

df["AveBedrms"] = df["total_bedrooms"] / df["households"]

df["AveOccup"] = df["population"] / df["households"]

# ==========================================
# 3. Select features
# ==========================================

X = df[
    [
        "median_income",
        "housing_median_age",
        "AveRooms",
        "AveBedrms",
        "population",
        "AveOccup",
        "latitude",
        "longitude"
    ]
]

y = df["median_house_value"]

# Rename columns to match screenshot
X.columns = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]

# ==========================================
# 4. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 5. Standard Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 6. Create models
# ==========================================

lr = LinearRegression()

ridge = Ridge(alpha=1.0)

lasso = Lasso(alpha=0.1)

elastic = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

# ==========================================
# 7. Train models
# ==========================================

lr.fit(X_train_scaled, y_train)

ridge.fit(X_train_scaled, y_train)

lasso.fit(X_train_scaled, y_train)

elastic.fit(X_train_scaled, y_train)

# ==========================================
# 8. Predictions
# ==========================================

y_pred_lr = lr.predict(X_test_scaled)

y_pred_ridge = ridge.predict(X_test_scaled)

y_pred_lasso = lasso.predict(X_test_scaled)

y_pred_elastic = elastic.predict(X_test_scaled)

# ==========================================
# 9. Model Evaluation
# ==========================================

def evaluate_model(name, y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n", name)
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)


evaluate_model(
    "Linear Regression",
    y_test,
    y_pred_lr
)

evaluate_model(
    "Ridge",
    y_test,
    y_pred_ridge
)

evaluate_model(
    "Lasso",
    y_test,
    y_pred_lasso
)

evaluate_model(
    "Elastic Net",
    y_test,
    y_pred_elastic
)

# ==========================================
# 10. Coefficient Comparison
# ==========================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Linear Regression": lr.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_,
    "Elastic Net": elastic.coef_
})

print("\n" + "=" * 75)
print(coefficients.to_string(index=True))
print("=" * 75)
