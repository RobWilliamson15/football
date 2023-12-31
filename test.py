"""
Trains and tests a model based on the scraped data from fbref
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
import numpy as np

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
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)
        return x

def train(X_train, X_test, y_train, y_test):
    """
    Trains and tests the Net model with the prepped data
    """
    # Initialize the model, loss function, and optimizer
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # Reduce LR by 10% every 10 epochs
    
    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    num_epochs = 150
    for epoch in range(num_epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.long)

        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(X_test, dtype=torch.float32)
            val_labels = torch.tensor(y_test, dtype=torch.long)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)

        # Append the loss values to the lists
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")
    
    # Plotting the Training and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    #plt.savefig("training_validation_loss.png")   # Save the figure to a PNG
    plt.show()

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

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    #plt.savefig("confusion_matrix.png")   # Save the figure to a PNG
    plt.show()

if __name__ == "__main__":
    TRAIN_DATA, TEST_DATA, TRAIN_LABELS, TEST_LABELS = data_prep('data/csv/cleaned_fixture_results.csv')
    train(TRAIN_DATA, TEST_DATA, TRAIN_LABELS, TEST_LABELS)
