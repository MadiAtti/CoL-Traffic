import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNetwork, self).__init__()
        
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