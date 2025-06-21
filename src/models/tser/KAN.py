import torch
from kan import KAN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class KANTrainer:
    def __init__(self, rnd_state, device):
        self.rnd_state = rnd_state
        self.device = device

    @staticmethod
    def _to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    def mse(self, dataset, model):
        y_true = self._to_numpy(dataset['test_label'])
        y_true = y_true.flatten().tolist()

        y_pred = self._to_numpy(model(dataset['test_input']).squeeze())
        y_pred = y_pred.flatten().tolist()

        return mean_squared_error(y_true, y_pred)
    
    def rmse(self, dataset, model):
        y_true = self._to_numpy(dataset['test_label'])
        y_true = y_true.flatten().tolist()

        y_pred = self._to_numpy(model(dataset['test_input']).squeeze())
        y_pred = y_pred.flatten().tolist()

        return np.sqrt(mean_squared_error(y_true, y_pred))

    def mae(self, dataset, model):
        y_true = self._to_numpy(dataset['test_label'])
        y_true = y_true.flatten().tolist()

        y_pred = self._to_numpy(model(dataset['test_input']).squeeze())
        y_pred = y_pred.flatten().tolist()

        return mean_absolute_error(y_true, y_pred)

    def r2(self, dataset, model):
        y_true = self._to_numpy(dataset['test_label'])
        y_true = y_true.flatten().tolist()

        y_pred = self._to_numpy(model(dataset['test_input']).squeeze())
        y_pred = y_pred.flatten().tolist()

        return r2_score(y_true, y_pred)

    def train_model(self, dataset, n, g, k):
        # 1) Preprocessar rótulos para regressão
        X_train = dataset['train_input'].to(self.device).float()
        y_train = dataset['train_label'].to(self.device).float().unsqueeze(1)

        X_test  = dataset['test_input'].to(self.device).float()
        y_test  = dataset['test_label'].to(self.device).float().unsqueeze(1)

        if X_train.dim() > 2:
            X_train = X_train.view(X_train.size(0), -1)
            X_test = X_test.view(X_test.size(0), -1)

        data = {
            'train_input':  X_train,
            'train_label':  y_train,
            'test_input':   X_test,
            'test_label':   y_test
        }

        # 2) Construir o modelo KAN para saída única
        input_dim = X_train.shape[1]
        model = KAN(
            width=[input_dim, *n, 1],
            grid=g, k=k,
            seed=self.rnd_state,
            device=self.device
        )

        # 3) Ajustar com MSELoss
        model.fit(
            data,
            opt="LBFGS",
            steps=100,
            loss_fn=torch.nn.MSELoss()
        )

        # 4) Retornar métricas
        return {
            'mse': self.mse(data, model),
            'mae': self.mae(data, model),
            'rmse': self.rmse(data, model),
            'r2': self.r2(data, model)
        }