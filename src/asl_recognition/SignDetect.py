import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
# import utils.MediaPipeUtilities as mpfs
from dotenv import load_dotenv
from datetime import datetime
# from utils.StopAtAccuracy import StopAtAccuracy
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, return_sequences = True):
        super().__init__()
        self.lstm=nn.LSTM(input_size, hidden_size, batch_first = True)
        self.return_sequences = return_sequences

    def forward(self, x):
        x, _ = self.lstm(x)
        return x if self.return_sequences else x[:, -1, :]

def Train():
    print('Training---------')
    load_dotenv()
    data_path = os.path.join(os.getenv('DATA_PATH'))
    no_sequences = int(os.getenv('no_sequences'))
    sequence_length = int(os.getenv('sequence_length'))
    actions = np.array(['hello', 'seeyoulater', 'I', 'father', 'mother', 'yes', 'no', 'help', 'please', 'thankyou', 'want', 'what', 'iloveyou'])
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(data_path, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    X = torch.tensor(np.array(sequences))
    y = F.one_hot(torch.tensor(labels), num_classes = 13).float()
    # print(y)
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

    model = nn.Sequential(
        LSTM(1662, 64, True),
        nn.ReLU(),
        LSTM(64, 128, True),
        nn.ReLU(),
        LSTM(128, 64, False),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, actions.shape[0])
    )

    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size = 30, shuffle = True, drop_last = True)
    test_loader = DataLoader(test_data, batch_size = test_data.tensors[0].shape[0])
    numepochs = 1

    for epochi in range(numepochs):
        for X,y in train_loader:
            yHat = model(X)
            loss = lossfun(yHat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            predicted_action = torch.argmax(yHat, axis = 1)
            print(predicted_action)
    
    torch.save(model,"bs_model.pth")
    return test_loader

def Evaluate(test_loader):
    model = torch.load("bs_model.pth", weights_only = False)
    model.eval()

    with torch.no_grad():
        X,y = next(iter(test_loader))
        yHat = model(X)

        predicted = torch.argmax(yHat, dim = 1)
        actual = torch.argmax(y, dim = 1)

        acc = (predicted == actual).float().mean()
        print(acc.item())


if __name__ == "__main__":
    t_loader = Train()
    Evaluate(t_loader)