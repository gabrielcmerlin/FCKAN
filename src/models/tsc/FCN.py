import torch.nn as nn
from torch.nn.functional import avg_pool1d

class FCN(nn.Module):
  def __init__(self, n_classes):
    
    super(FCN, self).__init__()
    self.block1 = nn.Sequential(
        nn.Conv1d(in_channels=1, 
                  out_channels=128,
                  kernel_size=8, 
                  padding='same'),  
        nn.BatchNorm1d(128),  
        nn.ReLU() 
    )
    
    self.block2 = nn.Sequential(
        nn.Conv1d(in_channels=128, 
                  out_channels=256, 
                  kernel_size=5,
                  padding='same'),  
        nn.BatchNorm1d(256),  
        nn.ReLU() 
    )
    
    self.block3 = nn.Sequential(
        nn.Conv1d(in_channels=256, 
                  out_channels=128, 
                  kernel_size=3,
                  padding='same'),  
        nn.BatchNorm1d(128),  
        nn.ReLU() 
    )
    
    self.fc = nn.Sequential(
        nn.Flatten(),  
        nn.Linear(128, n_classes),  
        nn.Softmax(dim=1) 
    )
  

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    # self.x_emb= avg_pool1d(x, x.shape[-1]) torch.saVE
    x = avg_pool1d(x, x.shape[-1])
    x = x.squeeze(-1)
    x = self.fc(x)
    return x