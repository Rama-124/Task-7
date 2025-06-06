

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('breast-cancer.csv')


# Inspect data
print("Dataset shape:", df.shape)
print(df.head())

# Prepare features and labels
X = df.drop(columns=['diagnosis'])  # assuming 'diagnosis' is target
y = df['diagnosis'].map({'M': 1, 'B': 0})  # Malignant = 1, Benign = 0

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train SVM with linear kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

# Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

# Visualization function
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

# Visualize decision boundaries
plot_decision_boundary(svm_linear, X_test, y_test, "SVM with Linear Kernel")
plot_decision_boundary(svm_rbf, X_test, y_test, "SVM with RBF Kernel")

# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.1, 1, 10]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, refit=True, cv=5)
grid.fit(X_scaled, y)

print("Best Parameters from GridSearch:", grid.best_params_)

# Cross-validation accuracy
scores = cross_val_score(grid.best_estimator_, X_scaled, y, cv=5)
print("Cross-validation Accuracy:", scores.mean())

# Final model evaluation
y_pred = grid.best_estimator_.predict(X_scaled)
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
