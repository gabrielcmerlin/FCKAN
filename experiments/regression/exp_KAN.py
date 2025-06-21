import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random
import torch
import pandas as pd
import time

from src.models.tser.KAN import KANTrainer
from src.utils.data_tser import DatasetManager
from experiments.regression.data_UCR import DATASETS

#Variables
EXPERIMENT_DATE = time.strftime("%Y%m%d_%H%M%S")
seed = RND_STATE = 42

RESULTS_DIR = 'results/'
NUMBER_OF_EXPERIMENTS = 1

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  
random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

results_data_dir = {
    'model': [],
    'dataset': [],
    'exp': [],
    'mse': [],
    'mae': [],
    'r2': [],
    'rmse': [],  # nova métrica
    'time': []
}

for dataset in DATASETS:
    print(f'Loading dataset {dataset}...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print('device:', device)
    dataset_manager = DatasetManager(dataset, device)

    print(dataset_manager.dataset['train_input'].shape)
    
    trainer = KANTrainer(RND_STATE, device)
    layers, gridsize, korder = [40, 40], 5, 3

    for experiment_number in range(NUMBER_OF_EXPERIMENTS):
        try:
            start_time = time.time()
            results = trainer.train_model(dataset_manager.dataset, n=layers, g=gridsize, k=korder)
            end_time = time.time()
        except Exception as e:
            msg = f'Error in dataset {dataset_manager.dataset_name}: {e}'
            file = open(f'{RESULTS_DIR}/errors/kan_error_log_{EXPERIMENT_DATE}.txt', 'a')
            file.write(f'{msg}\n')
            file.close()
            print(msg)
            continue

        results_data_dir['dataset'].append(dataset_manager.dataset_name)
        results_data_dir['model'].append('kan')
        results_data_dir['exp'].append(experiment_number)
        results_data_dir['mse'].append(results['mse'])
        results_data_dir['mae'].append(results['mae'])
        results_data_dir['rmse'].append(results['rmse'])
        results_data_dir['r2'].append(results['r2'])
        results_data_dir['time'].append(end_time - start_time)

        results_df = pd.DataFrame(results_data_dir)
        results_df.to_csv(f'{RESULTS_DIR}KAN_TSER_[40_40]_g5_k3_{EXPERIMENT_DATE}.csv', index=False)