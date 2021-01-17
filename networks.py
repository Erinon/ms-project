import torch


class DenseNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(DenseNetwork, self).__init__()

        self.linears = torch.nn.ModuleList()
        
        layer_sizes = [input_size] + layer_sizes
        for i, s in enumerate(layer_sizes[1:]):
            self.linears.append(torch.nn.Linear(layer_sizes[i], s))
            self.linears.append(torch.nn.ReLU())

        self.logits = torch.nn.Linear(layer_sizes[-1], output_size)

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
        
        x = self.logits(x)
        
        return x


class DuelingNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, adv_sizes, val_sizes,
                 output_size):
        super(DuelingNetwork, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        hidden_sizes = [input_size] + hidden_sizes
        for i, s in enumerate(hidden_sizes[1:]):
            self.hidden_layers.append(torch.nn.Linear(hidden_sizes[i], s))
            self.hidden_layers.append(torch.nn.ReLU())

        self.adv_layers = torch.nn.ModuleList()
        adv_sizes = [hidden_sizes[-1]] + adv_sizes
        for i, s in enumerate(adv_sizes[1:]):
            self.adv_layers.append(torch.nn.Linear(adv_sizes[i], s))
            self.adv_layers.append(torch.nn.ReLU())

        self.val_layers = torch.nn.ModuleList()
        val_sizes = [hidden_sizes[-1]] + val_sizes
        for i, s in enumerate(val_sizes[1:]):
            self.val_layers.append(torch.nn.Linear(val_sizes[i], s))
            self.val_layers.append(torch.nn.ReLU())

        self.adv_out = torch.nn.Linear(adv_sizes[-1], output_size)
        self.val_out = torch.nn.Linear(val_sizes[-1], 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        adv = x
        for layer in self.adv_layers:
            adv = layer(adv)
        adv = self.adv_out(adv)

        val = x
        for layer in self.val_layers:
            val = layer(val)
        val = self.val_out(val)

        x = val + adv - torch.mean(adv, dim=-1).unsqueeze(-1)
        
        return x


if __name__ == '__main__':
    input_size = 10
    hidden_sizes = [3, 4]
    adv_sizes = [5, 6]
    val_sizes = [7, 8]
    output_size = 9

    x = torch.autograd.Variable(torch.randn(32, input_size))

    dn = DuelingNetwork(input_size, hidden_sizes, adv_sizes, val_sizes,
                        output_size)

    out = dn(x)