import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = pd.read_csv("your_dataset.csv")  # Replace with actual dataset
df.ffill(inplace=True)

# Features and labels
X = df.drop('label_column', axis=1)  # Replace 'label_column' with your actual target column
y = df['label_column']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open("croprecommendation.pkl", "wb"))
