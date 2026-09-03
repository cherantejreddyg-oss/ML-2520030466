import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# The dataset 'df' is already loaded and preprocessed in the first cell.

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

# Create classification target
# 1 = expensive house, 0 = less expensive house
df["expensive"] = (
    df["median_house_value"] >= df["median_house_value"].median()
).astype(int)

# Features and target
X = df.drop(columns=["median_house_value", "expensive"])
y = df["expensive"]

# Handle missing values
X = X.fillna(X.median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 1. DECISION TREE - GINI
# ============================================================

dt_gini = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

dt_gini.fit(X_train, y_train)

y_pred_gini = dt_gini.predict(X_test)

gini_accuracy = accuracy_score(y_test, y_pred_gini)

print("\n================ GINI =================")
print("Gini Accuracy:", gini_accuracy)


# ============================================================
# 2. DECISION TREE - ENTROPY
# ============================================================

dt_entropy = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

dt_entropy.fit(X_train, y_train)

y_pred_entropy = dt_entropy.predict(X_test)

entropy_accuracy = accuracy_score(y_test, y_pred_entropy)

print("\n================ ENTROPY =================")
print("Entropy Accuracy:", entropy_accuracy)


# ============================================================
# 3. COST COMPLEXITY PRUNING (CCP)
# ============================================================

tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

tree.fit(X_train, y_train)

# Get pruning path
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Optimization: The list of alphas can be very long.
# We sample them to speed up execution while still capturing the trend.
ccp_alphas_sampled = ccp_alphas[::50]

train_scores = []
test_scores = []

print(f"Training {len(ccp_alphas_sampled)} trees for pruning analysis...")

for alpha in ccp_alphas_sampled:
    clf = DecisionTreeClassifier(
        criterion="gini",
        ccp_alpha=alpha,
        random_state=42
    )
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

# Best alpha from sampled list
best_alpha = ccp_alphas_sampled[np.argmax(test_scores)]
best_ccp_accuracy = max(test_scores)

print("\n================ CCP PRUNING =================")
print("Best Sampled CCP Alpha:", best_alpha)
print("Best CCP Test Accuracy:", best_ccp_accuracy)

# Plot CCP results
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas_sampled, train_scores, marker="o", label="Training Accuracy")
plt.plot(ccp_alphas_sampled, test_scores, marker="o", label="Testing Accuracy")
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.title("Cost Complexity Pruning (Sampled Alphas)")
plt.legend()
plt.grid()
plt.show()


# ============================================================
# 4. DEPTH SWEEP
# ============================================================

depths = range(1, 21)
depth_train_scores = []
depth_test_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )
    clf.fit(X_train, y_train)
    depth_train_scores.append(clf.score(X_train, y_train))
    depth_test_scores.append(clf.score(X_test, y_test))

best_depth = depths[np.argmax(depth_test_scores)]
best_depth_accuracy = max(depth_test_scores)

print("\n================ DEPTH SWEEP =================")
print("Best Depth:", best_depth)
print("Best Depth Test Accuracy:", best_depth_accuracy)

# Plot depth sweep
plt.figure(figsize=(10, 6))
plt.plot(depths, depth_train_scores, marker="o", label="Training Accuracy")
plt.plot(depths, depth_test_scores, marker="o", label="Testing Accuracy")
plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Depth Sweep")
plt.xticks(list(depths))
plt.legend()
plt.grid()
plt.show()

# ============================================================
# 5. FINAL COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Decision Tree - Gini",
        "Decision Tree - Entropy",
        "CCP Pruned Tree (Best Alpha)",
        "Best Depth Tree"
    ],
    "Accuracy": [
        gini_accuracy,
        entropy_accuracy,
        best_ccp_accuracy,
        best_depth_accuracy
    ]
})

print("\n================ FINAL COMPARISON =================")
print(results)
