import torch
from kan import KAN
from sklearn.metrics import f1_score, recall_score, precision_score

class KANTrainer:
    def __init__(self, n_classes, rnd_state, device):
            self.n_classes = n_classes
            self.rnd_state = rnd_state
            self.device = device

    @staticmethod
    def test_acc(dataset, model):
        y_pred = torch.argmax(model(dataset['test_input']), dim=1)
        y_true = dataset['test_label']

        return torch.mean((y_pred == y_true).float()).cpu().numpy()

    @staticmethod
    def f1(dataset, model):
        """
        Compute F1 score.
        """
        y_true = dataset['test_label']
        y_pred = torch.argmax(model(dataset['test_input']), dim=1)
        return f1_score(y_true.cpu().numpy() , y_pred.cpu().numpy() , average='weighted')

    @staticmethod
    def recall(dataset, model):
        """
        Compute recall score.
        """
        y_true = dataset['test_label']  # Ensure tensor is on CPU and convert to NumPy
        y_pred = torch.argmax(model(dataset['test_input']), dim=1)
        return recall_score(y_true.cpu().numpy() , y_pred.cpu().numpy() , average='weighted')

    @staticmethod
    def precision(dataset, model):
        """
        Compute precision score.
        """
        y_true = dataset['test_label'] 
        y_pred = torch.argmax(model(dataset['test_input']), dim=1)
        return precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy() , average='weighted')


    def train_model(self, dataset, n, g, k):
        input_dim = dataset['train_input'].shape[1]
        # n_outputs = len(torch.unique(dataset['train_label']))
        n_outputs = int(torch.max(dataset['train_label']).item()) + 1

        model = KAN(width=[input_dim, *n, n_outputs], 
                    grid=g, k=k, 
                    seed=self.rnd_state, device=self.device)
        
        model.fit(
            dataset,
            opt="LBFGS",
            steps=100,
            loss_fn=torch.nn.CrossEntropyLoss()
        )

        return {
            'acc': self.test_acc(dataset, model),
            'f1': self.f1(dataset, model),
            'recall': self.recall(dataset, model),
            'precision': self.precision(dataset, model)
        }