import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # for saving the model

# Load dataset
def load_data(filepath='soil_data.csv'):
    df = pd.read_csv(filepath)

    # Encode categorical data
    label_encoders = {}
    for column in ['Area', 'PlantType', 'MineralContent']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Split features and target
    X = df.drop('MineralContent', axis=1)
    y = df['MineralContent']
    return X, y, label_encoders

# Train model
def train_model():
    X, y, encoders = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("✅ Model Evaluation:\n", classification_report(y_test, y_pred))

    joblib.dump(clf, 'soil_model.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    print("✅ Model and encoders saved.")

# Predict new data
def predict_new(sensor_data):
    clf = joblib.load('soil_model.pkl')
    encoders = joblib.load('encoders.pkl')

    # Convert area and plant type to encoded form
    sensor_data['Area'] = encoders['Area'].transform([sensor_data['Area']])[0]
    sensor_data['PlantType'] = encoders['PlantType'].transform([sensor_data['PlantType']])[0]

    features = np.array([[sensor_data['Nitrogen'],
                          sensor_data['Phosphorus'],
                          sensor_data['Potassium'],
                          sensor_data['pH'],
                          sensor_data['Moisture'],
                          sensor_data['Area'],
                          sensor_data['PlantType']]])

    prediction = clf.predict(features)[0]
    mineral_content = encoders['MineralContent'].inverse_transform([prediction])[0]

    print(f"🧪 Predicted Mineral Content: {mineral_content}")

# Main
if __name__ == "__main__":
    import sys

    if not (os.path.exists('soil_model.pkl') and os.path.exists('encoders.pkl')):
        train_model()
    else:
        print("📦 Using trained model...")

    # Example sensor input (you can replace this with live sensor input later)
    example_input = {
        'Nitrogen': 60,
        'Phosphorus': 40,
        'Potassium': 50,
        'pH': 6.2,
        'Moisture': 25,
        'Area': 'Punjab',
        'PlantType': 'Wheat'
    }

    predict_new(example_input)
