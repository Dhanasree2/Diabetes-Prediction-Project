import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv('diabetes.csv')

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the dataset correctly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))
print("Model trained and saved successfully!")
