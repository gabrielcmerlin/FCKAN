import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
import time
import os

from tqdm import tqdm
from src.models.tser.FCN import FCNRegressor  # agora requer input_channels
from src.utils.data_tser import DatasetManager
from experiments.regression.data_UCR import DATASETS
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

EXPERIMENT_DATE = time.strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = 'results/'
NUMBER_OF_EXPERIMENTS = 1

# atualizar métricas para regressão
results_data_dir = {
    'model': [], 'dataset': [], 'exp': [],
    'mse': [], 'mae': [], 'r2': [], 'rmse': [],
    'best_train_loss': [], 'time': []
}

def FCNTrainer(model, train_loader, criterion, optimizer,
               num_epochs, path2bestmodel, dname, idx, device):
    """Treina no train_loader, salva o modelo de menor train-loss."""
    os.makedirs(path2bestmodel, exist_ok=True)
    model.to(device).train()
    print('device:', device)

    best_loss = float('inf')
    train_losses = []

    for epoch in tqdm(range(1, num_epochs+1), desc="Epoch"):
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # salva melhor modelo de acordo com train-loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       f"{path2bestmodel}/{dname}_run_{idx}_best.pth")

        print(f"[{epoch}/{num_epochs}] train-loss: {avg_loss:.4f}")

    return {'train_losses': train_losses, 'best_loss': best_loss}

def evaluate_on_test(model_class, checkpoint, input_channels, test_loader, device):
    """Carrega checkpoint e calcula métricas de regressão no test_loader."""
    model = model_class(input_channels).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds = out.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(y.numpy().flatten())

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    rmse = np.sqrt(mse)

    return {'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse}

# ─── loop principal ────────────────────────────────────────────────────────────
for dataset in DATASETS:
    print(f"\n=== Dataset {dataset} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datam = DatasetManager(name=dataset, device=device)
    train_loader, test_loader = datam.load_dataloader_for_training()

    # inferir canais de entrada
    first_batch = next(iter(train_loader))
    input_channels = first_batch[0].shape[1]

    for exp in range(NUMBER_OF_EXPERIMENTS):
        fcn = FCNRegressor(input_channels)  # passa numero de canais
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(fcn.parameters(),
                                     lr=0.001, betas=(0.9,0.999), eps=1e-8)
        start = time.time()

        # treina pelas num_epochs completas
        out = FCNTrainer(
            fcn, train_loader, criterion, optimizer,
            num_epochs=2000,
            path2bestmodel=f"weights/fcn_reg_{EXPERIMENT_DATE}",
            dname=datam.dataset_name,
            idx=exp,
            device=device
        )

        # avalia no test_loader
        ckpt = f"weights/fcn_reg_{EXPERIMENT_DATE}/{datam.dataset_name}_run_{exp}_best.pth"
        metrics = evaluate_on_test(FCNRegressor, ckpt, input_channels, test_loader, device)
        elapsed = time.time() - start

        # grava resultados
        results_data_dir['model'].append('fcn_reg')
        results_data_dir['dataset'].append(datam.dataset_name)
        results_data_dir['exp'].append(exp)
        results_data_dir['mse'].append(metrics['mse'])
        results_data_dir['mae'].append(metrics['mae'])
        results_data_dir['r2'].append(metrics['r2'])
        results_data_dir['rmse'].append(metrics['rmse'])
        results_data_dir['best_train_loss'].append(out['best_loss'])
        results_data_dir['time'].append(elapsed)

        pd.DataFrame(results_data_dir).to_csv(f"{RESULTS_DIR}FCN_REG_{EXPERIMENT_DATE}.csv", index=False)
    print("Done.")