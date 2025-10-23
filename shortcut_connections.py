import torch


layer_sizes=[3,3,3,3,3,1]

input=torch.tensor([1.0,0,-1.0])
torch.manual_seed(123)



class GELU_FN(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self,x):

        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (x+0.044715 * torch.pow(x,3))))
    

class DNN(torch.nn.Module):

    def __init__(self,layer_sizes,use_short):
        super().__init__()

        self.use_short=use_short

        self.layers=torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[0],layer_sizes[1]),GELU_FN()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[1],layer_sizes[2]),GELU_FN()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[2],layer_sizes[3]),GELU_FN()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[3],layer_sizes[4]),GELU_FN()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[4],layer_sizes[5]),GELU_FN()),
            ])
        
    def forward(self,x):

        for layer in self.layers:

            layer_out=layer(x)

            if self.use_short and x.shape==layer_out.shape:
              
                x = x + layer_out
            else:
                x=layer_out
        
        return x




def print_grads(model,x):

    # fwd pass
    
    out=model(x)
    target=torch.tensor([[0.0]])

    loss=torch.nn.MSELoss()
    loss=loss(out,target)

    # back pass
    loss.backward()

    for name,param in model.named_parameters():

        if 'weight' in name:

            print(f"{name} has grad mean {param.grad.abs().mean().item()}")



WOskip=DNN(layer_sizes,False)
Wskip=DNN(layer_sizes,True)
print_grads(WOskip,input)
print("-------------")
print_grads(Wskip,input)