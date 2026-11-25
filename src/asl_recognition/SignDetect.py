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
    X = np.array(sequences)
    y = F.one_hot(torch.tensor(labels), num_classes = 13)
    print(y)


if __name__ == "__main__":
    Train()