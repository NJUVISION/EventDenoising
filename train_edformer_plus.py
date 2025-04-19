import numpy as np
import torch
import os
import random
import argparse
from models.model_edformer_plus import EDformerPlus
from models.dataset import ED24
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import roc_curve, auc
import logging
from datetime import datetime
import shutil

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
    """Save the model and optimizer state."""
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
    parser.add_argument('--root', type=str, default='/workspace/shared/event_dataset/ECCV2024_datasets/ED24/', help='Training dataset path')
    parser.add_argument('--eval', type=str, default='/workspace/shared/event_dataset/ECCV2024_datasets/AUC_test/5hz/mix_result.txt', help='Evaluation dataset path')
    parser.add_argument('--output', type=str, default='./edformer_plus_training_logs', help='result path')
    parser.add_argument('--N', type=int, default=4096, help='Number of events')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--width', type=int, default=346, help='Width of events')
    parser.add_argument('--height', type=int, default=260, help='Height of events')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--pretrained', type=str, default='', help='Pretrained model path')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    Training_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(args.output, Training_time)
    os.makedirs(output_path, exist_ok=True)
    
    file_path = os.path.join(output_path, 'files')
    os.makedirs(file_path, exist_ok=True)
    shutil.copy('train_edformer_plus.py', file_path)
    shutil.copy('models/dataset.py', file_path)
    shutil.copy('models/model_edformer_plus.py', file_path)
    
    model_path = os.path.join(output_path, 'models')
    os.makedirs(model_path, exist_ok=True)

    writer = SummaryWriter(output_path)

    logging.basicConfig(filename=os.path.join(output_path, 'logging.txt'),
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    
    device = torch.device("cuda:0")

    logging.info('******************Data Loading*********************')
    dataset = ED24(args.root, args.N)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_events = pd.read_csv(args.eval, skiprows=1, delimiter=' ', dtype={'column1': np.int64, 'column2': np.int16, 'column3': np.int16, 'column4': np.int8}).values

    logging.info('******************Model Initialization*********************')
    model = EDformerPlus().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    start_epoch = 0
    if args.pretrained and os.path.exists(args.pretrained):
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Loaded pretrained model from {args.pretrained}, starting from epoch {start_epoch}.")
    else:
        logging.info("No pretrained model or path invalid. Starting training from scratch.")
        
    pre_roc = 0

    logging.info('******************Training*********************')
    for epoch in range(start_epoch, 60):
        total_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)

        writer.add_scalar('Train_Loss', total_loss, epoch)
        writer.add_scalar('Train_Accuracy', train_accuracy, epoch)

        logging.info(f'Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        save_model(epoch, model, optimizer, total_loss, f"{model_path}/model_{epoch}.pth")

        roc_auc = evaluate_model(model, eval_events, args)
        logging.info(f'Epoch {epoch+1}, ROC = {roc_auc:.4f}')
        writer.add_scalar('Eval_ROC', roc_auc, epoch)

        if roc_auc >= pre_roc:
            pre_roc = roc_auc
            logging.info(f'Epoch {epoch+1}, Best ROC = {roc_auc:.4f}')
            save_model(epoch, model, optimizer, total_loss, f"{model_path}/best_model.pth")

    writer.close()
