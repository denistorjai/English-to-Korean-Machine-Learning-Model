# Import Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import DatasetLoader

# Encoder, Encode Tokens 
class Encoder(nn.Module):
    def __init__(self, ModelInput, EmbeddedSize, HiddenSize, NumberofLayers, DropoutRate):
        super().__init__()

        # Set Class variables
        self.Embedding = nn.Embedding(ModelInput, EmbeddedSize)
        self.rnn = nn.LSTM(EmbeddedSize, HiddenSize, NumberofLayers, dropout=DropoutRate)
        self.dropout = nn.Dropout(DropoutRate)

    # encode forward function
    def forward(self, src):
        embedded = self.dropout(self.Embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        # return encode
        return hidden, cell
    
# Decode Tokens
class Decoder(nn.Module):
    def __init__(self, ModelOutput, EmbeddedSize, HiddenSize, NumberofLayers, DropoutRate):
        super().__init__()

        # Class variables
        self.Embedding = nn.Embedding(ModelOutput, EmbeddedSize)
        self.rnn = nn.LSTM(EmbeddedSize, HiddenSize, NumberofLayers, dropout=DropoutRate)
        self.fc_out = nn.Linear(HiddenSize, ModelOutput)
        self.dropout = nn.Dropout(DropoutRate)

    # decode forward function
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.Embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))

        # return decode
        return prediction, hidden, cell
    
