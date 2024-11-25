import pandas as pd
import numpy as np
import pickle
import os

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/Pawandeep-prog/deploy-ml-model-flask/refs/heads/master/iris.data')

# Process features and labels
X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y.reshape(-1))

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train, y_train)

# Specify output path
output_path = r'C:\Users\HP\Projects\Flask\Iris_ML_project\iri.pkl'

# Ensure the directory exists
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))

# Save the model
print(f"Saving model to: {output_path}")
pickle.dump(sv, open(output_path, 'wb'))

# Verify file creation
print("File exists:", os.path.exists(output_path))
