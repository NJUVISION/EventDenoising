import random
import torch
import pandas as pd
import numpy as np
import argparse
from models.model_enrr import ENRR
import tqdm

TIMESTAMP_COLUMN = 0
X_COLUMN = 1
Y_COLUMN = 2
POLARITY_COLUMN = 3
LABEL_COLUMN = 4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Inference(object):
    def __init__(self, model, seq_len) -> None:
        self.model = model
        self.seq_len = seq_len

    def inference(self, event_array):
        num_samples = len(event_array) // self.seq_len

        x = event_array[:, X_COLUMN] / 346
        y = event_array[:, Y_COLUMN] / 260
        polarity = event_array[:, POLARITY_COLUMN]
        timestamp = self.normalize_column(
            event_array[:, TIMESTAMP_COLUMN])
        event_array = np.hstack((timestamp.reshape(-1, 1), x.reshape(-1, 1),
                                 y.reshape(-1, 1), polarity.reshape(-1, 1)))
        event_array_reshaped = event_array[:num_samples *
                                           self.seq_len, :].reshape((num_samples, self.seq_len, 4))
        
        label_pred = []
        score = []
        
        for i in tqdm.tqdm(range(num_samples)):
            events_slice = event_array_reshaped[i, :, :]
            score_stacked =  self.process_slice(events_slice)
            score.append(score_stacked)

        scores = np.vstack(score)
        
        return scores

    def normalize_column(self, column):
        min_val = np.min(column)
        max_val = np.max(column)
        normalized_column = (column - min_val) / (max_val - min_val)
        return normalized_column

    def process_slice(self, events_slice):
        num = 1
        score_list = []
        processed_events = torch.tensor(events_slice).reshape(
            (1, events_slice.shape[0], events_slice.shape[1])).to(dtype=torch.float32).cuda()
        sub_sequence_size = self.seq_len // num
        for j in range(num):
            start_idx = j * sub_sequence_size
            end_idx = (j + 1) * sub_sequence_size
            sub_sequence = processed_events[:, start_idx:end_idx, :]
            with torch.no_grad():
                score = mod(sub_sequence)
            score_np = score.cpu().numpy()
            score_list.append(score_np)

        score_stacked = np.vstack(score_list)
        return score_stacked

def normalize_column(column):
    min_val = np.min(column)
    max_val = np.max(column)
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DND21 datasets.')
    parser.add_argument('-i', '--input_file',  type=str,
                        default='./data/DND21/5Hz/driving_mix_result.txt', help='path to load dataset')
    parser.add_argument('-m', '--model_path', type=str,
                        default='./pretrained/best_enrr.pth', help='path to model')
    args = parser.parse_args()

    setup_seed(42)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0") 

    mod = ENRR().cuda()
    mod.load_state_dict(torch.load(args.model_path, map_location=device))

    mod.eval()

    class_scores = {}

    event_file = args.input_file

    events = pd.read_csv(event_file, skiprows=1, delimiter=' ', dtype={
        'column1': np.int64, 'column2': np.int16, 'column3': np.int16, 'column4': np.int8})
    events = events.values

    model, seq_len = mod, 4096
    inference = Inference(model, seq_len)
    ENR = inference.inference(events)
    
    num_samples = len(events) // 4096
    event_tmp = events[:num_samples*4096]
    event_label = event_tmp[:,LABEL_COLUMN]
    
    print('Pred ENR: ', ENR.mean())
    print('GT ENR:', event_label.mean())