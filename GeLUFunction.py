import torch


class GELU_FN(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self,x):

        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (x+0.044715 * torch.pow(x,3))))
    

y= torch.tensor(torch.rand((2,3,768)))
gelu=GELU_FN()



class FeedForward(torch.nn.Module):

    def __init__(self,emb_dim):
        super().__init__()

        self.layers=torch.nn.Sequential(torch.nn.Linear(emb_dim,4*emb_dim),
                                        GELU_FN(),
                                        torch.nn.Linear(4*emb_dim,emb_dim))
    
    def forward(self,x):

        return self.layers(x)
    

ffn=FeedForward(y.shape[-1])
out=ffn(y)
print(out.shape)

