import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
import time
import os

from sklearn.metrics import confusion_matrix
import seaborn as sns

import json

from tqdm import tqdm
from src.models.tsc.MLP import MLP
from src.utils.data import DatasetManager
from experiments.classification.data_UCR import DATASETS
from torch import nn
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENT_DATE = time.strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = 'outputs/results/'
NUMBER_OF_EXPERIMENTS = 1

results_data_dir = {
    'model': [], 'dataset': [], 'exp': [],
    'acc': [], 'f1': [], 'recall': [], 'precision': [],
    'early_stopped_at': [], 'time': []
}

def MLPTrainer(model, train_loader, criterion, optimizer,
               num_epochs, path2bestmodel, dname, idx, device):
    
    os.makedirs(path2bestmodel, exist_ok=True)
    model.to(device).train()

    best_loss = float('inf')
    train_losses = []
    train_accs = []

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch"):
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = running_loss / len(train_loader)
        acc = correct / total
        train_losses.append(avg_loss)
        train_accs.append(acc)

        # salva melhor modelo de acordo com train-loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{path2bestmodel}/{dname}_run_{idx}_best.pth")

        print(f"[{epoch}/{num_epochs}] train-loss: {avg_loss:.4f}  train-acc: {acc:.4f}")

    return {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'epoch_end': num_epochs
    }

def evaluate_on_test(model, checkpoint, n_classes, test_loader, device):
    """Carrega checkpoint e calcula métricas no test_loader."""

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return {
        'acc': accuracy_score(all_labels, all_preds),
        'f1':  f1_score(all_labels, all_preds, average='weighted'),
        'recall':  recall_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'y_true': all_labels,
        'y_pred': all_preds
    }

# ─── loop principal ────────────────────────────────────────────────────────────
for dataset in DATASETS:
    print(f"\n=== Dataset {dataset} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datam = DatasetManager(name=dataset, device=device)
    n_classes = datam.get_classes_number()
    train_loader, test_loader = datam.load_dataloader_for_training()

    for x, y in train_loader:
        input_size = x.shape[2]
        break

    for exp in range(NUMBER_OF_EXPERIMENTS):
        model = MLP(input_size, n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.001, betas=(0.9,0.999), eps=1e-8)
        start = time.time()

        # 1) treina pelas num_epochs completas
        out = MLPTrainer(model, train_loader, criterion, optimizer,
                         num_epochs=100,
                         path2bestmodel=f"outputs/weights/MLP_tsc_{EXPERIMENT_DATE}",
                         dname=datam.dataset_name,
                         idx=exp,
                         device=device)

        with open(f'outputs/losses/tsc/{dataset}.json', 'w') as f:
            json.dump(out['train_loss'], f)
        
        # 2) avalia só no test_loader
        ckpt = f"outputs/weights/MLP_tsc_{EXPERIMENT_DATE}/{datam.dataset_name}_run_{exp}_best.pth"
        metrics = evaluate_on_test(model, ckpt, n_classes, test_loader, device)
        elapsed = time.time() - start

        cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {dataset}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f"outputs/conf_matrix/{dataset}.png")
        plt.close()

        # grava resultados
        results_data_dir['model'].append('mlp')
        results_data_dir['dataset'].append(datam.dataset_name)
        results_data_dir['exp'].append(exp)
        results_data_dir['acc'].append(metrics['acc'])
        results_data_dir['f1'].append(metrics['f1'])
        results_data_dir['recall'].append(metrics['recall'])
        results_data_dir['precision'].append(metrics['precision'])
        results_data_dir['early_stopped_at'].append(out['epoch_end'])
        results_data_dir['time'].append(elapsed)

        pd.DataFrame(results_data_dir).to_csv(f"{RESULTS_DIR}MLP_TSC_{EXPERIMENT_DATE}.csv", index=False)
    print("Done.")