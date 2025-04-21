import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10,
                        conv_filters=[16, 32, 64, 128, 256],
                        conv_kernel_size=3,
                        conv_activation=nn.ReLU(),
                        use_batchnorm=False,
                        dropout_rate=0.0,
                        dense_neurons=128,
                        dense_activation=nn.ReLU()):
        super(FlexibleCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        if isinstance(conv_kernel_size, int):
            conv_kernel_size = [conv_kernel_size] * 5
        elif len(conv_kernel_size) != 5:
            raise ValueError("conv_kernel_size must be an integer or a list of 5 integers.")

        in_c = in_channels
        for i in range(5):
            out_c = conv_filters[i]
            kernel_size = conv_kernel_size[i]
            layers = [nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(conv_activation)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if dropout_rate > 0 and i < 4:
                layers.append(nn.Dropout(dropout_rate))
            self.conv_layers.extend(layers)
            in_c = out_c

        self._to_linear = None
        self._determine_linear_input_size((1, in_channels, 224, 224))

        self.dense1 = nn.Linear(self._to_linear, dense_neurons)
        self.dense_activation = dense_activation
        if dropout_rate > 0:
            self.dropout_dense = nn.Dropout(dropout_rate)
        else:
            self.dropout_dense = nn.Identity()
        self.fc = nn.Linear(dense_neurons, num_classes)

    def _determine_linear_input_size(self, input_shape):
        x = torch.randn(input_shape)
        self.eval()
        with torch.no_grad():
            for layer in self.conv_layers:
                x = layer(x)
            self._to_linear = x.view(x.size(0), -1).shape[1]
        self.train()

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_dense(x)
        x = self.dense1(x)
        x = self.dense_activation(x)
        x = self.fc(x)
        return x