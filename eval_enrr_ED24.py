import random
import torch
import numpy as np
import argparse
from models.model_enrr import ENRR
import tqdm
import h5py

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

def read_h5_file(file_path):
    with h5py.File(file_path, 'r') as h5f:
        img_group = h5f['imgs']
        img_timestamps = np.array(img_group['timestamps'])
        img_data = {}
        for timestamp in img_timestamps:
            img_name = f'image_{timestamp}'
            if img_name in img_group:
                img_data[timestamp] = np.array(img_group[img_name])
        
        flow_group = h5f['flows']
        flow_timestamps = np.array(flow_group['timestamps'])
        flow_data = {}
        for timestamp in flow_timestamps:
            flow_name = f'flow_{timestamp}'
            if flow_name in flow_group:
                flow_data[timestamp] = np.array(flow_group[flow_name])

        event_group = h5f['events']
        events_1 = np.array(event_group['2.0'])
        events_2 = np.array(event_group['2.3'])
        events_3 = np.array(event_group['2.5'])
        events_4 = np.array(event_group['3.0'])
        events_5 = np.array(event_group['3.5'])
        
        return img_data, img_timestamps, flow_data, flow_timestamps, events_1, events_2, events_3, events_4, events_5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ED24 datasets.')
    parser.add_argument('-i', '--input_file',  type=str,
                        default='./data/ED24/car.h5', help='path to load dataset') # or ./data/ED24/school.h5
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

    img_data, img_timestamps, flow_data, flow_timestamps, events_1, events_2, events_3, events_4, events_5 = read_h5_file(event_file)
    
    #### Noise Ratio ###
    # events_2 3Hz/pixel
    # events_3 5Hz/pixel
    # events_4 7Hz/pixel
    # events_5 9Hz/pixel
    ####################
    
    events = events_5

    events_num = 4096

    model, seq_len = mod, events_num
    inference = Inference(model, seq_len)
    ENR = inference.inference(events)
    
    num_samples = len(events) // events_num
    event_tmp = events[:num_samples*events_num]
    event_label = event_tmp[:,LABEL_COLUMN]
    
    print('Pred ENR: ', ENR.mean())
    print('GT ENR:', event_label.mean())