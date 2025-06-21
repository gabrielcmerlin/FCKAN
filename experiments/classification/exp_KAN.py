import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random
import torch
import pandas as pd
import time

from src.models.tsc.KAN import KANTrainer
from src.utils.data import DatasetManager
from experiments.classification.data_UCR import DATASETS

#Variables
EXPERIMENT_DATE = time.strftime("%Y%m%d_%H%M")
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
    'acc': [],
    'f1': [],
    'recall': [],
    'precision': [],
    'time': []
}

for dataset in DATASETS:
    print(f'Loading dataset {dataset}...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print('device:', device)

    dataset_manager = DatasetManager(dataset, device)

    trainer = KANTrainer(dataset_manager.get_classes_number(), RND_STATE, device)
    layers, gridsize, korder = [40, 40], 5, 3

    for experiment_number in range(NUMBER_OF_EXPERIMENTS):
        try:
            start_time = time.time()
            results = trainer.train_model(dataset_manager.dataset, n=layers, g=gridsize, k=korder)
            end_time = time.time()
            print('--->', end_time - start_time)
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
        results_data_dir['acc'].append(results['acc'])
        results_data_dir['f1'].append(results['f1'])
        results_data_dir['recall'].append(results['recall'])
        results_data_dir['precision'].append(results['precision'])
        results_data_dir['time'].append(end_time - start_time)

        results_df = pd.DataFrame(results_data_dir)
        results_df.to_csv(f'{RESULTS_DIR}KAN_[40_40]_g5_k3_{EXPERIMENT_DATE}.csv', index=False)