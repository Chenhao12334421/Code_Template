# pylint: skip-file
from torchvision.models.resnet import resnet18, ResNet18_Weights
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class IdentityLayer(nn.Module):
    """
    An NN layer that returns the same input as output
    """
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x


class RecurrentResnet(nn.Module):
    def __init__(self, n_classes=7, rnn_hidden=256, rnn_layers=1):
        super(RecurrentResnet, self).__init__()

        # Define CNN feature extractor
        cnn = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        cnn.fc = IdentityLayer()
        self.cnn = cnn

        # Define LSTM
        self.lstm = nn.LSTM(512, rnn_hidden, rnn_layers, batch_first=True)

        # Define output layer
        self.fc = nn.Linear(rnn_hidden, n_classes)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape

        y = self.cnn(x[:, 0]).unsqueeze(1) # (batch_size, 1, feature_size)
        output, (hidden, cell) = self.lstm(y) # (batch_size, 1, hidden_size)
        for i in range(1, seq_len):
            y = self.cnn(x[:, i]).unsqueeze(1)
            output, (hidden, cell) = self.lstm(y, (hidden, cell))

        output = self.fc(output.squeeze()) # (batch_size, n_classes)
        return output


class RecurrentResnetWPacking(nn.Module):
    def __init__(self, n_classes=7, rnn_hidden=512, rnn_layers=1):
        super(RecurrentResnetWPacking, self).__init__()

        # Define CNN feature extractor
        cnn = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        # cnn = resnet34(ResNet34_Weights.IMAGENET1K_V1)
        cnn.fc = IdentityLayer()
        self.cnn = cnn

        # Define LSTM
        self.lstm = nn.LSTM(512, rnn_hidden, rnn_layers, batch_first=True)

        # Define output layer
        self.fc = nn.Linear(rnn_hidden, n_classes)

    def forward(self, x, actual_len):
        batch_size, seq_len, channels, height, width = x.shape

        # Extract CNN features (first have to flatten, then unflatten)
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)

        # Sort the x by real length, and pack the sequence into lstm then unpack
        _, sorted_indices = torch.sort(actual_len, descending=True)
        original_indices = torch.argsort(sorted_indices)
        x_sorted = x[sorted_indices]
        actual_len_sorted = actual_len[sorted_indices].cpu()
        actual_len_sorted = actual_len_sorted.to(torch.long)

        # Pack the sequences
        x_packed = pack_padded_sequence(x_sorted, actual_len_sorted, batch_first=True)

        # Send packed sequence to lstm
        x_packed, _ = self.lstm(x_packed)

        # Unpack the output
        x, output_lengths = pad_packed_sequence(x_packed, batch_first=True)
        x = x[torch.arange(x.size(0)), actual_len_sorted - 1]
        x = x[original_indices]
        x = self.fc(x)
        return x
