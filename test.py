"""
Trains and tests a model based on the scraped data from fbref
"""
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def data_prep(file_path):
    """
    Preps the data for training and testing
    """
    # Load the CSV data into a Pandas DataFrame
    data = pd.read_csv(file_path)

    # Preprocess the data
    encoder = LabelEncoder()
    data['Team1'] = encoder.fit_transform(data['Team1'])
    data['Team2'] = encoder.transform(data['Team2'])

    # Replace non-standard hyphen with standard hyphen
    data['FT'] = data['FT'].str.replace('–', '-')

    # Split the FT score column and create home_score and away_score columns
    data[['home_score', 'away_score']] = data['FT'].str.split('-', expand=True).astype(int)

    # Create a target variable (match outcome) as numerical labels (0 for Win, 1 for Draw, 2 for Lose)
    data['outcome'] = np.where(data['home_score'] > data['away_score'], 0,
                              np.where(data['home_score'] == data['away_score'], 1, 2))

    # Define features and target
    X = data[['Team1', 'Team2']].values
    y = data['outcome'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Define a simple feedforward neural network using PyTorch
class Net(nn.Module):
    """
    Defines a simple feedforward neural net
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # 3 output classes (Win, Draw, Lose)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(X_train, X_test, y_train, y_test):
    """
    Trains and tests the Net model with the prepped data
    """
    # Initialize the model, loss function, and optimizer
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 100
    for _ in range(num_epochs):
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

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_prep('data/cleaned_fixture_results.csv')
    train(X_train, X_test, y_train, y_test)
