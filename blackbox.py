# ------------------------------------------------------------------------------
#  HyperNOMAD - Hyper-parameter optimization of deep neural networks with
#               NOMAD.
#
#
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
#  for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#  You can find information on the NOMAD software at www.gerad.ca/nomad
# ------------------------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import os
import sys
from datahandler import DataHandler
from evaluator import *
from neural_net import NeuralNet
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
print('dimensione:', x_train.shape)
num_classes = len(np.unique(y_train))
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Read the inputs sent from HyperNOMAD
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Reading the inputs..')
# get the dataset
dataset = str(sys.argv[1])
# Architecture
num_conv_layers = int(sys.argv[2])

shift = 0
list_param_conv_layers = []
for i in range(num_conv_layers):
    conv_layer_param = (int(sys.argv[3 + shift]), int(sys.argv[4 + shift]), int(sys.argv[5 + shift]),
                        int(sys.argv[6 + shift]), int(sys.argv[7 + shift]))
    list_param_conv_layers += [conv_layer_param]
    shift += 5

last_index = shift + 2
num_full_layers = int(sys.argv[last_index + 1])
list_param_full_layers = []
for i in range(num_full_layers):
    list_param_full_layers += [int(sys.argv[last_index + 2 + i])]

# First 2 : blackbox.py, dataset
batch_size_index = 2 + (2 + num_conv_layers*5 + num_full_layers)
batch_size = int(sys.argv[batch_size_index])

# HPs
optimizer_choice = int(sys.argv[batch_size_index + 1])
arg1 = float(sys.argv[batch_size_index + 2])               # lr
arg2 = float(sys.argv[batch_size_index + 3])               # momentum
arg3 = float(sys.argv[batch_size_index + 4])               # weight decay
arg4 = float(sys.argv[batch_size_index + 5])               # dampening
dropout_rate = float(sys.argv[batch_size_index + 6])
activation = int(sys.argv[batch_size_index + 7])

# Load the data
print('> Preparing the data..')

if dataset != 'CUSTOM':
    dataloader = DataHandler(dataset, batch_size)
    #commento le righe successive perchè non sto classificando immagini
    image_size, number_classes = dataloader.get_info_data
    trainloader, validloader, testloader = dataloader.get_loaders()
else:
    # Add here the adequate information
    #image_size = None
   # number_classes = None
   # trainloader = None
   # validloader = None
   # testloader = None
    train_features = torch.Tensor(x_train)
    train_labels = torch.Tensor(y_train).long()
    test_features = torch.Tensor(x_test)
    test_labels = torch.Tensor(y_test).long()
    valid_features = torch.Tensor(x_valid)
    valid_labels = torch.Tensor(y_valid).long()

    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    valid_dataset=TensorDataset(valid_features, valid_labels)
    batch_size = 32 
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validloader=validloader= DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    image_size = (x_train.shape[1],1, 1)  # Numero di canali di input, lunghezza, larghezza
    number_classes = num_classes
# Test if the correct information is passed - especially in the case of CUSTOM dataset
assert isinstance(trainloader, torch.utils.data.dataloader.DataLoader), 'Trainloader given is not of class DataLoader'
assert isinstance(validloader, torch.utils.data.dataloader.DataLoader), 'Validloader given is not of class DataLoader'
assert isinstance(testloader, torch.utils.data.dataloader.DataLoader), 'Testloader given is not of class DataLoader'
assert image_size is not None, 'Image size can not be None'
assert number_classes is not None, 'Total number of classes can not be None'

num_input_channels = image_size[0] #modifica rispetto alla trasposta
list_param_conv_layers=[(6, 1, 1, 0, 1), (16, 1, 1, 0, 1)] #devo metterlo così per aver kernel 1 altrimenti mi da errore
print('> Constructing the network')
# construct the network
print('gli output sono',num_conv_layers, num_full_layers, list_param_conv_layers, list_param_full_layers,
                dropout_rate, activation, image_size[1], number_classes, num_input_channels,'fine')
cnn = NeuralNet(num_conv_layers, num_full_layers, list_param_conv_layers, list_param_full_layers,
                dropout_rate, activation, image_size[1], number_classes, num_input_channels)

cnn.to(device)

try:
    if optimizer_choice == 1:
        optimizer = optim.SGD(cnn.parameters(), lr=arg1, momentum=arg2, weight_decay=arg3,
                              dampening=arg4)
    if optimizer_choice == 2:
        optimizer = optim.Adam(cnn.parameters(), lr=arg1, betas=(arg2, arg3), weight_decay=arg4)
    if optimizer_choice == 3:
        optimizer = optim.Adagrad(cnn.parameters(), lr=arg1, lr_decay=arg2, weight_decay=arg4,
                                  initial_accumulator_value=arg3)
    if optimizer_choice == 4:
        optimizer = optim.RMSprop(cnn.parameters(), lr=arg1, momentum=arg2, alpha=arg3, weight_decay=arg4)
except ValueError:
    print('optimizer got an empty list')
    exit(0)

print(cnn)
print({'device':device, 'cnn':cnn, 'trainloader':trainloader, 'validloader':validloader, 'testloader':testloader, 'optimizer':optimizer, 'batch_size':batch_size, 'dataset':dataset})
# The evaluator trains and tests the network
evaluator = Evaluator(device, cnn, trainloader, validloader, testloader, optimizer, batch_size, dataset)
print('> Training')
print(evaluator)
best_val_acc, best_epoch = evaluator.train()
print('> Testing')
test_acc = evaluator.test()

# Output of the blackbox
print('> Final accuracy %.3f' % test_acc)
