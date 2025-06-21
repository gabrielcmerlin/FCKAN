import torch.nn as nn
from kan import KAN
from torch.nn.functional import avg_pool1d

class FCKAN(nn.Module):
  def __init__(self, input_channels, rnd_state, device):
    
    super(FCKAN, self).__init__()
    self.block1 = nn.Sequential(
        nn.Conv1d(in_channels=input_channels, 
                  out_channels=128, 
                  kernel_size=8),  
        nn.BatchNorm1d(128),  
        nn.ReLU() 
    )
    
    self.block2 = nn.Sequential(
        nn.Conv1d(in_channels=128, 
                  out_channels=256, 
                  kernel_size=5),  
        nn.BatchNorm1d(256),  
        nn.ReLU() 
    )
    
    self.block3 = nn.Sequential(
        nn.Conv1d(in_channels=256, 
                  out_channels=128, 
                  kernel_size=3),  
        nn.BatchNorm1d(128),  
        nn.ReLU() 
    )
    
    self.kan = KAN(width=[128, 40, 1], 
                   grid=5, k=3, seed=rnd_state, device=device)
  
  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)

    self.feature_extractor = avg_pool1d(x, x.shape[-1])
    self.feature_extractor = self.feature_extractor.squeeze(-1)

    self.pred_kan = self.kan(self.feature_extractor)

    return self.pred_kan