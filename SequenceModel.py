# Import Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import DatasetLoader

# Encoder, Encode English Tokens 
class Encoder(nn.Module):
    def __init__(self, ModelInput, EmbeddedSize, HiddenSize, NumberofLayers, DropoutRate):
        super().__init__()
        self.Embedding = nn.Embedding(ModelInput, EmbeddedSize)
        