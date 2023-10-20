import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV data into a Pandas DataFrame
data = pd.read_csv('data/cleaned_fixture_results.csv')

# Preprocess the data
encoder = LabelEncoder()
data['Team1'] = encoder.fit_transform(data['Team1'])
data['Team2'] = encoder.transform(data['Team2'])

# Replace non-standard hyphen with standard hyphen
data['FT'] = data['FT'].str.replace('–', '-')

# Split the FT score column and create home_score and away_score columns
data[['home_score', 'away_score']] = data['FT'].str.split('-', expand=True).astype(int)

# Create a target variable (match outcome) as numerical labels (0 for Win, 1 for Draw, 2 for Lose)
data['outcome'] = np.where(data['home_score'] > data['away_score'], 0, np.where(data['home_score'] == data['away_score'], 1, 2))

# Define features and target
X = data[['home_score', 'away_score']].values
y = data['outcome'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple feedforward neural network using PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 3)  # 3 output classes (Win, Draw, Lose)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Making predictions
with torch.no_grad():
    test_inputs = torch.tensor(X_test, dtype=torch.float32)
    predictions = model(test_inputs)
    predicted_labels = np.argmax(predictions, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, predicted_labels)
report = classification_report(y_test, predicted_labels)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

