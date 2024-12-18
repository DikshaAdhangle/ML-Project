import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = pd.read_csv('collected_data/hand_landmarks.csv')

# Split into features (X) and labels (y)
X = data.drop(columns=['label'])
y = data['label']

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with standard scaling and SVM classifier
model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'gesture_model.pkl')

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
