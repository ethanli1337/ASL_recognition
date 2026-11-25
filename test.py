import os
import numpy as np

data_path = 'src/asl_recognition/data'
no_sequences = 30
sequence_length = 30
skip = {'hello', 'father'}          # keep real recordings

for action in sorted(os.listdir(data_path)):
    action_dir = os.path.join(data_path, action)
    if not os.path.isdir(action_dir) or action in skip:
        continue
    for sequence in range(no_sequences):
        seq_dir = os.path.join(action_dir, str(sequence))
        os.makedirs(seq_dir, exist_ok=True)
        for frame_num in range(sequence_length):
            keypoints = np.random.rand(1662).astype(np.float32)   # fake frame
            np.save(os.path.join(seq_dir, str(frame_num)), keypoints)
    print(f'{action}: filled')