import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Load dataset
df = pd.read_csv("WinePredictor.csv")

#Features and labels
X = df.drop("Class", axis=1)
Y = df["Class"]

#Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train)

#Predict the model
Y_pred = model.predict(X_test)

print("Predicted classes:")
print(Y_pred)

#Accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)