import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('Crop_recommendation.csv')

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

print("Training models... Please wait.")
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# 6. Make predictions for all models
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred) * 100
svm_acc = accuracy_score(y_test, svm_pred) * 100
knn_acc = accuracy_score(y_test, knn_pred) * 100

print(f"\n--- Benchmarking Results ---")
print(f"Random Forest Accuracy: {rf_acc:.2f}%")
print(f"SVM Accuracy:         {svm_acc:.2f}%")
print(f"KNN Accuracy:         {knn_acc:.2f}%\n")

print("--- Random Forest Classification Report ---")
print(classification_report(y_test, rf_pred))

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf_model.classes_,
            yticklabels=rf_model.classes_)

plt.title('Confusion Matrix - Random Forest Crop Prediction', fontsize=16)
plt.xlabel('Predicted Crop', fontsize=12)
plt.ylabel('Actual Crop', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('confusion_matrix.png')
print("Saved 'confusion_matrix.png'")
plt.show()

models = ['Random Forest', 'SVM', 'KNN']
accuracies = [rf_acc, svm_acc, knn_acc]

plt.figure(figsize=(10, 6))
# Create the bar chart
sns.barplot(x=models, y=accuracies, palette='viridis')

plt.title('Model Benchmarking: Accuracy Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Machine Learning Algorithms', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.ylim(0, 110)

for i, v in enumerate(accuracies):
    plt.text(i, v + 2, f"{v:.2f}%", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()

plt.savefig('model_benchmarking_chart.png')
print("Saved 'model_benchmarking_chart.png'\n")
plt.show()

new_data = pd.read_csv('new_soil_data.csv')
X_new = new_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

predictions = rf_model.predict(X_new)
new_data['Recommended_Crop'] = predictions

print("--- Crop Recommendations for New Soil Data ---")
print(new_data[['N', 'P', 'K', 'Recommended_Crop']])
