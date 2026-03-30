import torch.nn as nn

class TrafficNN(nn.Module):
    '''
    Simple feedforward neural network for traffic classification.
    The architecture consists of two hidden layers with ReLU activations.
    The number of neurons in the hidden layers is determined by a common 
    heuristic based on the input dimension and the number of classes.
    '''
    def __init__(self, input_dim, num_classes):
        super(TrafficNN, self).__init__()
        
        hidden_layer1_neurons = int((2 / 3) * input_dim + num_classes)
        hidden_layer2_neurons = input_dim
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_layer1_neurons),
            nn.ReLU(),
            nn.Linear(hidden_layer1_neurons, hidden_layer2_neurons),
            nn.ReLU(),
            nn.Linear(hidden_layer2_neurons, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)