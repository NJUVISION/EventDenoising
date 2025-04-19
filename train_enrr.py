import numpy as np
import torch
import os
import random
import argparse
from models.model_enrr import ENRR
from models.dataset import ED24
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import shutil
import pandas as pd
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
        
        event_array = np.hstack((timestamp.reshape(-1, 1), x.reshape(-1, 1),
                                  y.reshape(-1, 1), polarity.reshape(-1, 1)))
        event_array_reshaped = event_array[:num_samples * self.seq_len].reshape((num_samples, self.seq_len, 4))
        
        scores = np.vstack([self.process_slice(event_array_reshaped[i]) for i in tqdm.tqdm(range(num_samples))])
        
        return scores

    def normalize_column(self, column):
        min_val = np.min(column)
        max_val = np.max(column)
        return (column - min_val) / (max_val - min_val)

    def process_slice(self, events_slice):
        processed_events = torch.tensor(events_slice, dtype=torch.float32).unsqueeze(0).cuda()
        sub_sequence_size = self.seq_len

        with torch.no_grad():
            score = self.model(processed_events)

        return score.cpu().numpy()


def save_model(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def train_one_epoch(model, train_loader, loss, optimizer, device):
    model.train()
    total_loss = 0

    for events, labels in tqdm.tqdm(train_loader):
        events, labels = events.to(device), labels.to(device)
        outputs = model(events)

        label_score = labels.squeeze(-1).mean(-1).unsqueeze(-1)
        label_score = label_score.to(device)
        score_loss = loss(outputs, label_score)

        optimizer.zero_grad()
        score_loss.backward()
        optimizer.step()

        total_loss += score_loss.item()

    return total_loss

def evaluate_model(model, eval_events, args):
    model.eval()
    inference = Inference(model, args)
    score = inference.inference(eval_events)
    return score


if __name__ == "__main__":
    setup_seed(230086)

    parser = argparse.ArgumentParser(description='The training code of ENRR')
    parser.add_argument('--root', type=str, default='/workspace/shared/event_dataset/ECCV2024_datasets/ED24/', help='Training dataset path')
    parser.add_argument('--eval', type=str, default='/workspace/shared/event_dataset/ECCV2024_datasets/AUC_test/5hz/mix_result.txt', help='Evaluation dataset path')
    parser.add_argument('--output', type=str, default='./enrr_training_logs', help='result path')
    parser.add_argument('--N', type=int, default=4096, help='Number of events')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--width', type=int, default=346, help='Width of events')
    parser.add_argument('--height', type=int, default=260, help='Height of events')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
    parser.add_argument('--pretrained', type=str, default='', help='Pretrained model path')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    Training_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(args.output, Training_time)
    os.makedirs(output_path, exist_ok=True)
    
    file_path = os.path.join(output_path, 'files')
    os.makedirs(file_path, exist_ok=True)
    shutil.copy('train_enrr.py', file_path)
    shutil.copy('models/dataset.py', file_path)
    shutil.copy('models/model_enrr.py', file_path)
    
    model_path = os.path.join(output_path, 'models')
    os.makedirs(model_path, exist_ok=True)

    writer = SummaryWriter(output_path)
    
    logging.basicConfig(filename=os.path.join(output_path, 'logging.txt'),
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

    device = torch.device("cuda:0")

    logging.info('****************** Data Loading *********************')
    dataset = ED24(args.root, args.N)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    eval_events = pd.read_csv(args.eval, skiprows=1, delimiter=' ', dtype={'column1': np.int64, 'column2': np.int16, 'column3': np.int16, 'column4': np.int8, 'column5': np.int8}).values
    
    num_samples = len(eval_events) // args.N
    event_tmp = eval_events[:num_samples*args.N]
    event_label = event_tmp[:,LABEL_COLUMN]
    mean_label = event_label.mean()

    logging.info('****************** Model Initialization *********************')
    model = ENRR(width=args.width, height=args.height).to(device)
    MSE_loss = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    start_epoch = 0
    if args.pretrained and os.path.exists(args.pretrained):
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Loaded pretrained model from {args.pretrained}, starting from epoch {start_epoch}.")
    else:
        logging.info("No pretrained model or path invalid. Starting training from scratch.")
        
    pre_delta_score = np.inf
    
    logging.info('****************** Training *********************')
    for epoch in range(start_epoch, 100):
        total_loss = train_one_epoch(model, train_loader, MSE_loss, optimizer, device)

        writer.add_scalar('Train_Loss', total_loss, epoch)
        
        save_model(epoch, model, optimizer, total_loss, f"{model_path}/model_{epoch}.pth")

        score = evaluate_model(model, eval_events, args)
        
        mean_score = score.mean()
        delta_score = np.abs(mean_label-mean_score)
        
        logging.info(f'Epoch {epoch+1}, delta_score = {delta_score:.4f}, mean_label = {mean_label:.4f}, mean_score = {mean_score:.4f}')
        writer.add_scalar('delta_score', delta_score, epoch)

        if delta_score <= pre_delta_score:
            pre_delta_score = delta_score
            logging.info(f'Epoch {epoch+1}, Best delta_score = {delta_score:.4f}')
            save_model(epoch, model, optimizer, total_loss, f"{model_path}/best_model.pth")

