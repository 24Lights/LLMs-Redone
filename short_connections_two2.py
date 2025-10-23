import torch

# Deeper network to see the effect more clearly
deep_layer_sizes = [3] + [3]*10 + [1]  # 20 hidden layers
input = torch.tensor([1.0, 0, -1.0])
torch.manual_seed(123)

class GELU_FN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (x+0.044715 * torch.pow(x,3))))

class DNN(torch.nn.Module):
    def __init__(self, layer_sizes, use_short):
        super().__init__()
        self.use_short = use_short
        
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]), 
                    GELU_FN()
                )
            )
        
    def forward(self, x):
        for layer in self.layers:
            layer_out = layer(x)
            if self.use_short and x.shape == layer_out.shape:
                x = x + layer_out
            else:
                x = layer_out
        return x

def print_grads(model, x, name):
    # Zero gradients first
    model.zero_grad()
    
    # Forward pass
    out = model(x)
    target = torch.tensor([[0.0]])
    
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(out, target)

    # Backward pass
    loss.backward()

    print(f"\n{name}:")
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name and param.grad is not None:
            layer_idx = i // 2  # Since each layer has weight and bias
            print(f"Layer {layer_idx}: grad mean = {param.grad.abs().mean().item():.6f}")

# Test with deeper network
deep_WOskip = DNN(deep_layer_sizes, False)
deep_Wskip = DNN(deep_layer_sizes, True)

print("="*60)
print("DEEP NETWORK COMPARISON (20 layers)")
print("="*60)

print_grads(deep_WOskip, input, "WITHOUT Skip Connections")
print_grads(deep_Wskip, input, "WITH Skip Connections")