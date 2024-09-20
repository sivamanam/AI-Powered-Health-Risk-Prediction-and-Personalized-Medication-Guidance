import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv(r"C:\Users\Lenovo\Downloads\now jmc\dataset1.csv")

# Apply Label Encoding to the 'Gender' column (if needed)
label_encoder = LabelEncoder()
data["Gender"] = label_encoder.fit_transform(data["Gender"])
#Blood Pressure
data["Blood Pressure"] = label_encoder.fit_transform(data["Blood Pressure"])
data["Genetic Marker for Diabetes"] = label_encoder.fit_transform(data["Genetic Marker for Diabetes"])


# Prepare input (X) and output (y) data
x = data.iloc[:, :-1].values  # Features (all columns except the last one)
y = data.iloc[:, -1].values   # Target (last column)

# Split the data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=2000)

# Initialize and train the Naive Bayes model
modelnb = GaussianNB()
modelnb.fit(xtrain, ytrain)

# Predict on the test set
ypred = modelnb.predict(xtest)

# Calculate the accuracy of the model
accuracy = accuracy_score(ytest, ypred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model using joblib
joblib.dump(modelnb, 'naive_bayes_model1.pkl')
print("Model saved successfully!")
