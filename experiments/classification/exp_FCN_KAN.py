import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
import time
import os

from kan import KAN

from tqdm import tqdm
from src.models.tsc.FCN import FCN
from src.utils.data import DatasetManager
from experiments.classification.data_UCR import DATASETS
from torch import nn
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import numpy as np

EXPERIMENT_DATE = time.strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = 'results/'
NUMBER_OF_EXPERIMENTS = 1

results_data_dir = {
    'model': [], 'dataset': [], 'exp': [],
    'acc': [], 'f1': [], 'recall': [], 'precision': [],
    'early_stopped_at': [], 'time': []
}

def FCNTrainer(model, train_loader, criterion, optimizer,
               num_epochs, path2bestmodel, dname, idx, device):
    """Treina só no train_loader, salva o modelo de menor train-loss, sem early stopping."""
    
    os.makedirs(path2bestmodel, exist_ok=True)
    model.to(device).train()

    best_loss = float('inf')
    train_losses = []
    train_accs = []

    for epoch in tqdm(range(1, num_epochs+1), desc="Epoch"):
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

def evaluate_on_test(model_class, checkpoint, n_classes, test_loader, device):
    """Carrega checkpoint e calcula métricas no test_loader."""
    model = model_class(n_classes).to(device)
    model.fc = KAN(width=[128, 40, n_classes], 
                grid=5, k=3, seed=42, device=device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
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
        'precision': precision_score(all_labels, all_preds, average='weighted')
    }

# ─── loop principal ────────────────────────────────────────────────────────────
for dataset in DATASETS:
    print(f"\n=== Dataset {dataset} ===")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    datam = DatasetManager(name=dataset, device=device)
    n_classes = datam.get_classes_number()
    train_loader, test_loader = datam.load_dataloader_for_training()

    for exp in range(NUMBER_OF_EXPERIMENTS):
        fcn = FCN(n_classes)
        fcn.load_state_dict(torch.load(f'weights/[END] tsc_fcn/{dataset}_run_0_best.pth'))
        fcn.fc = KAN(width=[128, 40, n_classes], 
                   grid=5, k=3, seed=42, device=device)

        for param in fcn.block1.parameters():
            param.requires_grad = False
        for param in fcn.block2.parameters():
            param.requires_grad = False
        for param in fcn.block3.parameters():
            param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(fcn.parameters(),
                                     lr=0.001, betas=(0.9,0.999), eps=1e-8)
        start = time.time()

        # 1) treina pelas num_epochs completas
        out = FCNTrainer(fcn, train_loader, criterion, optimizer,
                         num_epochs=100,
                         path2bestmodel=f"weights/fcn_kan_{EXPERIMENT_DATE}",
                         dname=datam.dataset_name,
                         idx=exp,
                         device=device)

        # 2) avalia só no test_loader
        ckpt = f"weights/fcn_kan_{EXPERIMENT_DATE}/{datam.dataset_name}_run_{exp}_best.pth"
        metrics = evaluate_on_test(FCN, ckpt, n_classes, test_loader, device)
        elapsed = time.time() - start

        # grava resultados
        results_data_dir['model'].append('fcn')
        results_data_dir['dataset'].append(datam.dataset_name)
        results_data_dir['exp'].append(exp)
        results_data_dir['acc'].append(metrics['acc'])
        results_data_dir['f1'].append(metrics['f1'])
        results_data_dir['recall'].append(metrics['recall'])
        results_data_dir['precision'].append(metrics['precision'])
        results_data_dir['early_stopped_at'].append(out['epoch_end'])
        results_data_dir['time'].append(elapsed)

        pd.DataFrame(results_data_dir).to_csv(f"{RESULTS_DIR}FCN_KAN_{EXPERIMENT_DATE}.csv", index=False)
    print("Done.")