import random
import torch
import numpy as np
import argparse
from models.model_edformer_plus import EDformerPlus
import h5py
import os

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

class Inference:
    def __init__(self, model, args, seq_len):
        self.model = model
        self.seq_len = seq_len
        self.width = args.width
        self.height = args.height

    def inference(self, event_array):
        num_samples = len(event_array) // self.seq_len
        x = event_array[:, X_COLUMN] / self.width
        y = event_array[:, Y_COLUMN] / self.height
        polarity = event_array[:, POLARITY_COLUMN]
        timestamp = self.normalize_column(event_array[:, TIMESTAMP_COLUMN])
        
        event_array = np.column_stack((timestamp, x, y, polarity))
        event_array_reshaped = event_array[:num_samples * self.seq_len].reshape(num_samples, self.seq_len, 4)

        label_pred = np.vstack([self.process_slice(event_slice) for event_slice in event_array_reshaped])
        return label_pred

    def normalize_column(self, column):
        min_val = np.min(column)
        max_val = np.max(column)
        return (column - min_val) / (max_val - min_val)

    def process_slice(self, events_slice):
        first_column = events_slice[:, 0]
        events_slice[:, 0] = self.normalize_column(first_column)

        processed_events = torch.tensor(events_slice, dtype=torch.float32).unsqueeze(0).cuda()
        with torch.no_grad():
            f = self.model(processed_events)
            predictions = torch.sigmoid(f).squeeze(0).cpu().numpy()
        return predictions
        
def calculate_model_size(model):
    return sum(p.numel() for p in model.parameters()) * 4    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DVSCLEAN datasets.')
    parser.add_argument('-i', '--input_folder',  type=str,
                        default='/workspace/shared/event_dataset/DVSCLEAN/simulated_data/', help='path to load dataset folder')
    parser.add_argument('-m', '--model_path', type=str,
                        default='./pretrained/best_edformer_plus.pth', help='path to model')
    parser.add_argument('--width', type=int, default=1280, help='Width of events')
    parser.add_argument('--height', type=int, default=720, help='Height of events')
    args = parser.parse_args()
    
    np.set_printoptions(suppress=True, precision=12)
    setup_seed(42)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0") 
    
    event_files = [os.path.join(args.input_folder, file) for file in os.listdir(args.input_folder)]
    event_files_50 = [file for file in event_files if '_50' in os.path.basename(file)]
    event_files_100 = [file for file in event_files if '_100' in os.path.basename(file)]
    
    SNR = []
    for event_file in event_files_100:
        print(event_file)
        with h5py.File(event_file, 'r') as hdf:
            t = hdf['/events/timestamp'][:]
            x = hdf['/events/x'][:]
            y = hdf['/events/y'][:]
            p = hdf['/events/polarity'][:]
            p = np.where(p < 0.5, 0, 1)
            label = hdf['/events/label'][:]
            events = np.vstack((t,x,y,p)).T

        mod = EDformerPlus().cuda()
        mod.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        mod.eval()
        
        # model_size = calculate_model_size(mod)
        # print(f'EDformer++ Model size: {model_size / (1024**2):.2f} MB')
        
        if '_50' in event_file:
            N = len(events) // 4
        elif '_100' in event_file:
            N = len(events) // 5
        model, seq_len = mod, N
        inference = Inference(model, args, N)

        label_pred_stacked = inference.inference(events)
        
        num_samples = len(events) // N
        event_tmp = events[:num_samples * N, :]
        label = label[:num_samples * N]
        label = label.reshape(-1,1)
        print(events.shape)
        
        indices = np.where(label_pred_stacked < 0.01)[0]
        binary_array = np.ones_like(label_pred_stacked)
        binary_array[indices] = 0
        res_array = event_tmp[indices]
        
        true_signals = np.sum(label == 0) 
        predicted_signals = np.sum(binary_array == 0)  

        true_positive = np.sum((binary_array == 0) & (label == 0)) 
        false_positive = np.sum((binary_array == 0) & (label == 1)) 

        if false_positive > 0:
            snr = 10 * np.log10(true_positive / false_positive)
        else:
            snr = np.inf
        print("Pred SNR:", snr)
        SNR.append(snr)

    print(f"Mean SNR:", np.mean(SNR))