import pickle
import numpy as np

model = pickle.load(
    open("models/model.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

sample_customer = np.array([
    [1, 50, 2000]
])

sample_customer = scaler.transform(
    sample_customer
)

prediction = model.predict(
    sample_customer
)

print(prediction)
