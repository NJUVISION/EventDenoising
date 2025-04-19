import numpy as np
import torch
import os
import random
import argparse
from models.model_edformer_plus import EDformerPlus
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import roc_curve, auc
import logging

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
    def __init__(self, model, args):
        self.model = model
        self.seq_len = args.N
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
            predictions = torch.sigmoid(self.model(processed_events)).squeeze(0).cpu().numpy()
            
        return predictions

def save_model(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for events, labels in tqdm(train_loader):
        events, labels = events.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(events)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) >= 0.01).float()
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.numel()

    accuracy = correct_predictions / total_samples
    return total_loss, accuracy

def evaluate_model(model, eval_events, args):
    model.eval()
    inference = Inference(model, args)
    label_pred_stacked = inference.inference(eval_events)
    
    num_samples = len(eval_events) // args.N
    event_tmp = eval_events[:num_samples * args.N]
    event_labels = event_tmp[:, LABEL_COLUMN].reshape(-1, 1)

    fpr, tpr, _ = roc_curve(event_labels, label_pred_stacked)
    
    return auc(fpr, tpr)

if __name__ == "__main__":
    setup_seed(230086)

    parser = argparse.ArgumentParser(description='EDformer++')
    parser.add_argument('-i', '--input_file', type=str, default='./data/DND21/5Hz/driving_mix_result.txt', help='Evaluation dataset path')
    parser.add_argument('--N', type=int, default=4096*7, help='Number of events')
    parser.add_argument('--width', type=int, default=346, help='Width of events')
    parser.add_argument('--height', type=int, default=260, help='Height of events')
    parser.add_argument('-m', '--model_path', type=str, default='./pretrained/best_edformer_plus.pth', help='Pretrained model path')
    args = parser.parse_args()
    
    device = torch.device("cuda:0")
    eval_events = pd.read_csv(args.input_file, skiprows=1, delimiter=' ', dtype={'column1': np.int64, 'column2': np.int16, 'column3': np.int16, 'column4': np.int8}).values
    print(eval_events.shape)

    model = EDformerPlus().to(device)
    
    if args.model_path and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logging.info("No pretrained model or path invalid.")
        
    AUC = evaluate_model(model, eval_events, args)
    print(f'AUC = {AUC:.4f}')