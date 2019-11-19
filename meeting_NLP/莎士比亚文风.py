import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.embedding