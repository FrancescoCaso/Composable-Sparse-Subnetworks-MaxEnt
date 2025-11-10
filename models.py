import torch
import torch.nn as nn
import torch
from torch.functional import F

import numpy as np
from pathlib import PosixPath

def binarize_weights(tensor):
    return torch.where(tensor >= 0, torch.tensor(1.0), torch.tensor(-1.0))

class BinarizationLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Definisci il primo layer
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        # Definisci il secondo layer
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Primo passaggio: applica il primo layer
        x = self.linear1(x)
        # Secondo passaggio: applica il secondo layer
        x = self.linear2(x)
        return x

def save_weights_and_biases(model):
    return {
        'fc1_weights': model.fc1.weight.detach().clone().data.numpy(),
        'fc1_biases': model.fc1.bias.detach().clone().data.numpy(),
        'fc2_weights': model.fc2.weight.detach().clone().data.numpy(),
        'fc2_biases': model.fc2.bias.detach().clone().data.numpy(),
        'fc3_weights': model.fc3.weight.detach().clone().data.numpy(),
        'fc3_biases': model.fc3.bias.detach().clone().data.numpy(),
    }

def create_model(model_name, params):
    models = {
        "MLP": MLP,
        "ConvNet": ConvNet
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(models.keys())}")
    
    model_class = models[model_name]
    model = model_class(**params)
    
    return model

def load_model(params: dict, 
               model_: str = './', 
               model_name: str = 'MLP'):
    model = create_model(model_name, params)
    if type(model_) in [str, PosixPath]:
        model_ = torch.load(model_, weights_only=False)['model_state_dict']
    elif type(model_) != dict:
        raise ValueError("model_ must be a dictionary or a path to a model.")
    
    model.load_state_dict(model_)
    return model

def extract_weights_and_biases(model):
    weights_and_biases = {}
    for i, layer in enumerate(model.layers):
        weights_and_biases[f'fc{i+1}_weights'] = layer.weight.detach().clone().data.numpy()
        weights_and_biases[f'fc{i+1}_biases'] = layer.bias.detach().clone().data.numpy()
    return weights_and_biases

class ConvNet(nn.Module):
    def __init__(self, 
                 input_channels: int = 3, 
                 input_dim: int = 28*28, 
                 hidden_conv_dims: list = [16, 8], 
                 hidden_fc_dims: list = [128, 10], 
                 output_dim: int = 2, 
                 dropout_rate: float = 0.3, 
                 activation_fn: nn.Module = nn.ReLU, 
                 conv_kernel_size: int = 3, 
                 padding: int = 1, 
                 conv_stride: int = 1, 
                 pool_kernel_size: int = 2, 
                 pool_stride: int = 2,
                 t = 1.0,
                 **kwargs):
        '''
        Initializes the ConvNet.

        Args:
            input_channels (int): The number of input channels.
            input_dim (int): The number of input features.
            hidden_conv_dims (list of int): List of hidden layer dimensions for Convolution layers.
            hidden_fc_dims (list of int): List of hidden layer dimensions for fully connected layers.
            output_dim (int): The number of output features.
            dropout_rate (float): The rate of dropout.
            activation_fn (nn.Module): The activation function to use (e.g., nn.ReLU()).
            kernel_size (int): size of the convolution kernel.
            padding (int): number of padding.
        '''
        super(ConvNet, self).__init__()

        self.name = "ConvNet"
        self.input_dim = input_dim
        self.conv_layers = nn.ModuleList()
        self.conv_activations = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_activations = nn.ModuleList()
        self.dropout = nn.Dropout(p = dropout_rate)
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
        self.t = t
        

        prev_dim = input_channels
        final_dim = np.sqrt(input_dim)
        for hidden_dim in hidden_conv_dims:
            self.conv_layers.append(nn.Conv2d(prev_dim, hidden_dim, kernel_size = conv_kernel_size, padding = padding))
            self.conv_activations.append(activation_fn())
            prev_dim = hidden_dim
            final_dim = ( final_dim - conv_kernel_size + 2 * padding ) / conv_stride + 1
            final_dim = ( final_dim - pool_kernel_size ) / pool_stride + 1
            final_dim = np.floor(final_dim)

        prev_dim = int( hidden_conv_dims[-1] * final_dim * final_dim )
        for hidden_dim in hidden_fc_dims:
            self.fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.fc_activations.append(activation_fn())
            prev_dim = hidden_dim

        self.fc_layers.append(nn.Linear(prev_dim, output_dim, bias=False))
        self.representations = {}

    def forward(self, x):
        self.representations['input'] = x.detach().clone()
        x = x.view(-1, 1, 28, 28)  # se immagini in scala di grigi

        for i, (layer, activation) in enumerate(zip(self.conv_layers, self.conv_activations)):
            pre_activation = layer(x)
            self.representations[f'conv{i+1}_pre_activation'] = pre_activation.detach().clone()

            x = activation(pre_activation)
            self.representations[f'conv{i+1}_post_activation'] = x.detach().clone()

            x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        for i, (layer, activation) in enumerate(zip(self.fc_layers[:-1], self.fc_activations)):
            pre_activation = layer(x)
            self.representations[f'fc{i+1}_pre_activation'] = pre_activation.detach().clone()

            x = activation(pre_activation)
            self.representations[f'fc{i+1}_post_activation'] = x.detach().clone()

            x = self.dropout(x)

        output = self.fc_layers[-1](x)
        self.representations[f'fc{len(self.fc_layers)}_pre_activation'] = output.detach().clone()

        # output = F.softmax(output, dim=1)
        output = output/self.t
        self.representations[f'fc{len(self.fc_layers)}_post_activation'] = output.detach().clone()

        return output
        
class MLP(nn.Module):
    def __init__(self, 
                 input_dim: int = 28*28, 
                 hidden_dims: list = [16, 8], 
                 output_dim: int = 2, 
                 t: float = 7.0,
                 activation_fn: nn.Module = nn.ReLU, 
                 positivity_constraint: bool = False, 
                 binary: bool = False, 
                 mask: torch.tensor=None,
                 dropout: float = 0.3,
                 bias_last: bool = False,
                 **kwargs):
        """
        Initializes the MLP.
        
        Args:
            input_dim (int): The number of input features.
            hidden_dims (list of int): List of hidden layer dimensions.
            output_dim (int): The number of output features.
            dropout_rate (float): The rate of dropout.
            activation_fn (nn.Module): The activation function to use (e.g., nn.ReLU()).
        """
        super(MLP, self).__init__()
        
        # self.name = "MLP"
        self.input_dim = input_dim
        self.dropout = nn.Dropout(p=dropout)
        self.name = "MLP"
        # Create a list to hold the layers
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        # activation_fn = nn.ReLU()
        # activation_fn = nn.ReLU()
        # Add the first layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.activations.append(activation_fn())
            prev_dim = hidden_dim
        if positivity_constraint:
            self.weight_transform = torch.nn.ReLU()
        elif binary:
            self.weight_transform = binarize_weights
        else:
            self.weight_transform = torch.nn.Identity()
        
        self.output = nn.Linear(prev_dim, output_dim, bias=bias_last)
        self.layers.append(self.output)
        
        self.representations = {}
        self.t = t
        
        
        self.mask_list=[]
        start=0
        for i in hidden_dims+[output_dim]:
            end = start+i
            if mask is not None:
                self.mask_list.append(mask[start:end])
            else:
                self.mask_list.append(None)
            start=start+i


    def block_neurons(self, x, mask):
        if mask is not None:
            x = x * mask
        return x
    
    def forward(self, x):
        try:
            x = x.view(-1, self.input_dim)
        except:
            raise ValueError(f"Change the input shape, expected {self.input_dim} but got {x.size(1)}")
        self.representations['input'] = x.detach().clone()
        for i, (layer, activation) in enumerate(zip(self.layers[:-1], self.activations)):
            # weight = self.weight_transform(layer.weight)
            # bias = self.weight_transform(layer.bias)
            # pre_activation = F.linear(x, weight, bias)
            pre_activation = layer(x)
            self.representations[f'fc{i+1}_pre_activation'] = pre_activation.detach().clone()

            x = activation(pre_activation)
            self.representations[f'fc{i+1}_post_activation'] = x.detach().clone()

            # x = self.block_neurons(x, mask=self.mask_list[i])
            x = self.dropout(x)

        # output = self.layers[-1](x)
        output = self.output(x)
        self.representations[f'fc{len(self.layers)}_pre_activation'] = output.detach().clone()
        
        output = output/self.t
        # out_max , _ = output.max(dim=1, keepdim=True)
        # out_max = output.max()
        # output = output - out_max
        # output = torch.exp(output)
        # output = output / output.sum(dim=1, keepdim=True)
        # output = F.softmax(output, dim=1)
        # self.representations[f'fc{len(self.layers)}_post_activation'] = output.detach().clone()
        
        return output
    
    def expand_output_layer(self, new_classes):
        old_weights = self.output.weight.data
        old_bias = self.output.bias.data
        old_out_features = self.output.out_features
        in_features = self.output.in_features

        new_out_features = old_out_features + new_classes
        new_output = nn.Linear(in_features, new_out_features)

        # Copy old weights and bias
        new_output.weight.data[:old_out_features] = old_weights
        new_output.bias.data[:old_out_features] = old_bias

        # Replace the old output layer
        self.output = new_output
    

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
        self.representations = {'x':0,
                                'fc1_pre_activation':0, 'fc1_post_activation':0,
                                'fc2_pre_activation':0, 'fc2_post_activation':0,
                                'fc3':0, 'output':0}

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        self.representations['x'] = x.detach().clone()
        x = self.fc1(x)
        
        self.representations['fc1_pre_activation'] = x.detach().clone()
        x = torch.relu(x)
        
        self.representations['fc1_post_activation'] = x.detach().clone()
        x = self.fc2(x)
        
        self.representations['fc2_pre_activation'] = x.detach().clone()
        x = torch.relu(x)
        
        self.representations['fc2_post_activation'] = x.detach().clone()
        x = self.fc3(x)
        
        self.representations['fc3'] = x.detach().clone()
        x = F.softmax(x, dim=1)
        
        self.representations['output'] = x.detach().clone()
        return x

if __name__ == "__main__":
    from train_utils import get_mnist_loaders, eval
    classes = [0, 1]
    train_loader, val_loader, test_loader = get_mnist_loaders(classes = classes, batch_size = 64)
    
    mask = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.])
    model_ablated=load_model(model_='./models_01/final_models/0.pth', mask=mask)
    accuracy, cm = eval(model_ablated, 0, test_loader)
    print(cm)
        