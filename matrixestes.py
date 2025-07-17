import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Import classifiers
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Load and clean data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Remove inconsistent vectors
expected_len = len(data[0])
filtered_data = [d for d in data if len(d) == expected_len]
filtered_labels = [l for d, l in zip(data, labels) if len(d) == expected_len]

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Define all classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP Neural Network": MLPClassifier(max_iter=1000),
    "AdaBoost": AdaBoostClassifier()
}

results = {}
f1_scores = []

# Train and evaluate each model
for name, model in classifiers.items():
    print(f"\n🔍 Training: {name}")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm
    }

    f1_scores.append(f1)

    print(f"✅ Accuracy       : {acc*100:.2f}%")
    print(f"✅ Precision      : {prec:.4f}")
    print(f"✅ Recall         : {rec:.4f}")
    print(f"✅ F1 Score       : {f1:.4f}")
    print(f"🧮 Confusion Matrix:\n{cm}")
    print(f"📊 Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

# Save best model
best_model_name = max(results, key=lambda x: results[x]['f1_score'])
best_model = results[best_model_name]['model']
with open('best_model.p', 'wb') as f:
    pickle.dump({'model': best_model}, f)

print(f"\n🎯 Best Model: {best_model_name} (F1 Score: {results[best_model_name]['f1_score']:.4f})")
print("💾 Model saved as: best_model.p")

# 📊 Plot only Random Forest's F1 Score
rf_f1 = results["Random Forest"]['f1_score']

plt.figure(figsize=(6, 4))
plt.bar(["Random Forest"], [rf_f1], color='mediumslateblue')
plt.ylabel("F1 Score")
plt.title("Random Forest Model (F1 Score)")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 📈 Plot only Random Forest's Confusion Matrix
rf_cm = results["Random Forest"]['confusion_matrix']

plt.figure(figsize=(6, 5))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
