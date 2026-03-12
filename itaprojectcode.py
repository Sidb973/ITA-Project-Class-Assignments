import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Crop_recommendation.csv')

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

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
plt.show()

new_data = pd.read_csv('new_soil_data.csv')

X_new = new_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

predictions = rf_model.predict(X_new)

new_data['Recommended_Crop'] = predictions

print("--- Crop Recommendations for New Soil Data ---")
print(new_data[['N', 'P', 'K', 'Recommended_Crop']])
