import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from spikingjelly.activation_based import spike_op as sn
from spikingjelly.activation_based import functional, surrogate, neuron, layer, encoding


class CSNN(nn.Module):
    def __init__(self, beta=0.9, slope=25):
        super(CSNN, self).__init__()

        # Convolutional layers with 12 filters in first layer, 64 filters in second, etc.
        self.conv1 = nn.Conv2d(1, 32, 5)  # Input is 1 channel (grayscale)
        self.lif1 = snn.Leaky(
            beta=beta, spike_grad=surrogate.fast_sigmoid(slope=slope))

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.lif2 = snn.Leaky(
            beta=beta, spike_grad=surrogate.fast_sigmoid(slope=slope))

        # Adjusted for smaller output size after convolution and pooling
        self.fc1 = nn.Linear(64 * 9*9, 128)
        self.lif3 = snn.Leaky(
            beta=beta, spike_grad=surrogate.fast_sigmoid(slope=slope))

        self.fc2 = nn.Linear(128, 7)  # 7 classes for FER2013 emotions
        self.lif4 = snn.Leaky(
            beta=beta, spike_grad=surrogate.fast_sigmoid(slope=slope))

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        # Forward pass through convolution and pooling layers
        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)
        # spk1 = self.dropout1(spk1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)
        # spk2 = self.dropout2(spk2)

        cur3 = self.fc1(spk2.view(spk2.size(0), -1))
        spk3, mem3 = self.lif3(cur3, mem3)
        # spk3 = self.dropout3(spk3)

        out = self.fc2(spk3)
        spk4, mem4 = self.lif4(out, mem4)

        return out, spk4, mem4


class FeedforwardSNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FeedforwardSNN, self).__init__()
        self.fc1 = nn.Linear(48 * 48, 512)    # First fully connected layer
        # Leaky integrate-and-fire layer for spiking
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(512, 256)        # Second fully connected layer
        self.lif2 = snn.Leaky(beta=0.9)
        self.fc3 = nn.Linear(256, 128)  # Output layer
        self.lif3 = snn.Leaky(beta=0.9)
        self.fc4 = nn.Linear(128, num_classes)  # Output layer
        self.lif4 = snn.Leaky(beta=0.9)

    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        # print(x.size())
        # Layer 1 with spiking
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        # print(spk1.size())
        # Layer 2 with spiking
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        # print(spk2.size())
        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)

        out = self.fc4(spk3)
        # print(spk3.size())

        return out, spk3, mem3


class SJCSNN(nn.Module):
    def __init__(self, T=8):
        super(SJCSNN, self).__init__()
        self.T = T
        self.layer1 = nn.Sequential(
            layer.Conv2d(1, 128, kernel_size=3, stride=1,
                         padding=1, step_mode='s'),
            layer.BatchNorm2d(128, step_mode='s'),
        )
        self.layer2 = nn.Sequential(
            neuron.IFNode(v_threshold=1.0,
                          surrogate_function=surrogate.ATan(), step_mode='m'),
            layer.MaxPool2d(kernel_size=2, stride=2),
            layer.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            layer.BatchNorm2d(128),
            neuron.IFNode(v_threshold=1.0,
                          surrogate_function=surrogate.ATan(), step_mode='m'),
            layer.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            layer.Flatten(),
            layer.Dropout(p=0.5),
            layer.Linear(128*12*12, 1152),
            neuron.LIFNode(tau=2.0, v_reset=0.0,
                           surrogate_function=surrogate.ATan(), step_mode='m'),
            layer.Dropout(p=0.5),
            layer.Linear(1152, 128),
            neuron.LIFNode(tau=2.0, v_reset=0.0,
                           surrogate_function=surrogate.ATan(), step_mode='m'),
            layer.Linear(128, 7),
            neuron.LIFNode(tau=2.0, v_reset=0.0,
                           surrogate_function=surrogate.ATan(), step_mode='m')
        )

        functional.set_step_mode(self.layer2, step_mode='m')
        functional.set_step_mode(self.layer3, step_mode='m')
        functional.set_backend(self, backend='cupy')
        self.pe = encoding.PoissonEncoder()

    def forward(self, x):
        # x = self.pe(x)
        x = self.layer1(x)
        x_step = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out1 = self.layer2(x_step)
        out2 = self.layer3(out1)
        # for t in range(self.T):
        #     x_time[t] = self.pe(x)
        # x_step = []
        # for t in range(self.T):
        #     out1 = self.layer2(x_time[t])
        #     out2 = self.layer3(out1)
        #     x_step.append(out2.unsqueeze(0))
        # x = torch.cat(x_step)
        return out2.mean(dim=0)
