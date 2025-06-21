import torch.nn as nn

class FCN(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.layers = nn.ModuleList([
            # First convolutional layer.
            nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Second convolutional layer.
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Third convolutional layer.
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Global Average Pooling and Flattening.
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten()
        ])
    
    def forward(self, x):
        # Processing every layer in the FCN and then returning.
        for layer in self.layers:
            x = layer(x)

        return x
    
class FCNRegressor(FCN):

    def __init__(self, input_channels):
        super().__init__(input_channels)
        self.final_layer = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = super().forward(x)
        x = self.final_layer(x)

        return x