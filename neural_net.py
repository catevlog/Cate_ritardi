# ' ------------------------------------------------------------------------------
#  HyperNOMAD - Hyper-parameter optimization of deep neural networks with
# '             NOMAD.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.environ.get('HYPERNOMAD_HOME')+"/src/blackbox/blackbox")


class NeuralNet(nn.Module):
    def __init__(self, num_conv_layers, num_full_layers, list_param_conv_layers, list_param_full_layers, dropout_rate,
                 activation, initial_image_size, total_classes, number_input_channels):
        """
            Initialize a CNN.
            We suppose that the initial image size is 32.
        """

        super(NeuralNet, self).__init__()

        self.activation = activation
        self.dropout = dropout_rate

        self.init_im_size = initial_image_size
        self.total_classes = total_classes
        self.in_size_first_full_layer = -1
        self.num_conv_layers = num_conv_layers
        self.num_full_layers = num_full_layers
        self.param_conv = list_param_conv_layers
        self.param_full = list_param_full_layers
        self.number_input_channels = number_input_channels

        assert num_conv_layers == len(list_param_conv_layers), 'len(list_param_conv_layers) != num_conv_layers'
        for i in range(num_conv_layers):
            assert len(list_param_conv_layers[i]) == 5, 'Problem with number of parameters of the convolutional layer '\
                                                        'num %r' % i
            assert num_full_layers == len(list_param_full_layers), 'num_full_layers != len(list_param_full_layers)'

        self.features, self.classifier = self.construct_network()

    def construct_network(self):
        """
        Construct a CNN.

        list_param_conv_layers = [(n_out_channel, kernel_size, stride, padding, do_pool)]
        list_param_full_layers = [n_output_layer1,...]
        """
        layers = []
        n_in_channel = self.number_input_channels
        # construct the convolutional layers
        for i in range(self.num_conv_layers):
            params_i = self.param_conv[i]
            n_out_channel = params_i[0]
            kernel_size = params_i[1]
            stride = params_i[2]
            padding = params_i[3]

            layer = nn.Conv1d(n_in_channel, n_out_channel, kernel_size, stride=stride,
                                 padding=padding,bias=False)
            
            nn.init.xavier_uniform_(layer.weight)
            layers += [layer]
            if self.activation == 1:
                layers += [nn.ReLU(inplace=True)]
            if self.activation == 2:
                layers += [nn.Sigmoid()]
            if self.activation == 3:
                layers += [nn.Tanh()]
            layers += [nn.Dropout1d(self.dropout)]
            layers += [nn.BatchNorm1d(n_out_channel)]

            pooling = params_i[-1]                                                                                              
            # if params_i[-1] == 1:                                                                                             
            layers += [nn.MaxPool1d(kernel_size=pooling, stride=pooling)]                                                       
                                                                                                                                
            n_in_channel = n_out_channel                                                                                        
        if self.num_conv_layers > 0:
            layers += [nn.AvgPool1d(kernel_size=1, stride=1)]                                                                       
        self.features = nn.Sequential(*layers)                                                                                  
        layers = []                                                                                                             
        # print('Done with conv layer')                                                                                         
                                                                                                                                
        # construct the full layers                                                                                             
        if not self.param_conv:                                                                                                 
            size_input = self.number_input_channels * (self.init_im_size ** 2)                                                  
        else:                                                                                                                   
            size_input = (self.get_input_size_first_lin_layer() ** 2) * self.param_conv[-1][0]                                  
                                                                                                                                
        self.in_size_first_full_layer = size_input                                                                              
                                                                                                                                
        for i in range(self.num_full_layers):                                                                                   
            layer = nn.Linear(size_input, self.param_full[i])                                                                   
            nn.init.xavier_uniform_(layer.weight)                                                                               
            layers += [layer]                                                                                                   
            size_input = self.param_full[i]                                                                                     
            if self.activation == 1:                                                                                            
                layers += [nn.ReLU(inplace=True)]                                                                               
            if self.activation == 2:                                                                                            
                layers += [nn.Sigmoid()]                                                                                        
            if self.activation == 3:                                                                                            
                layers += [nn.Tanh()]                                                                                           
            layers += [nn.Dropout(self.dropout)]                                                                                
            layers += [nn.BatchNorm1d(self.param_full[i])]                                                                      
                                                                                                                                
        layer = nn.Linear(size_input, self.total_classes)                                                                       
        nn.init.xavier_uniform_(layer.weight)                                                                                   
        layers += [layer]                                                                                                       
        layers += [nn.BatchNorm1d(self.total_classes)]                                                                          
        self.classifier = nn.Sequential(*layers)                                                                                
        return self.features, self.classifier   

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(-1, self.in_size_first_full_layer)
        x = self.classifier(x)
        return x

    def get_input_size_first_lin_layer(self):                                                                                   
        """                                                                                                                     
        :return: current_size                                                                                                   
        """                                                                                                                     
        current_size = self.init_im_size                                                                                        
        for i in range(self.num_conv_layers):                                                                                   
            temp = (current_size - self.param_conv[i][1] + 2 * self.param_conv[i][3]) / self.param_conv[i][2] + 1               
            current_size = np.floor(temp)                                                                                       
            # if self.param_conv[i][-1] == 1:                                                                                   
            #     if current_size > 1:                                                                                          
            #         current_size = np.floor(current_size / 2)                                                                 
                                                                                                                                
            # Pooling                                                                                                           
            current_size = np.floor(current_size / self.param_conv[i][-1])                                                      
            current_size = int(current_size)
            print(current_size)                                                                                   
        return current_size  
