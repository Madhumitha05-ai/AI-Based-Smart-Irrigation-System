import joblib
import numpy as np

model = joblib.load('irrigation_model.pkl')

input_data = np.array([[40, 32, 45, 2, 2]])
prediction = model.predict(input_data)

print("💧 Water Required (liters):", round(prediction[0], 2))
