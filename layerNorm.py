import torch
from torch import nn

torch.set_printoptions(sci_mode=False)
x= torch.rand((2,5))

layer=torch.nn.Sequential(torch.nn.Linear(5,6),torch.nn.ReLU())
output=layer(x)

mean=output.mean(dim=-1,keepdim=True)
var=output.var(dim=-1,keepdim=True)

print("x is ",x)
print("output is ",output)
print("mean :",mean)
print("var",var)

output_norm=(output-mean )/torch.sqrt(var)

print("mean after layer norm is ",output_norm.mean(dim=-1,keepdim=True))
print("variance after layer norm is ",output_norm.var(dim=-1,keepdim=True))


class LayerNorm(nn.Module):

    def __init__(self,emb_dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.ones(emb_dim))
        self.eps=1e-5
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False )

        normalised=(x-mean)/torch.sqrt(var+self.eps)

        return self.scale*normalised+self.shift

ln=LayerNorm(5)
normalised=ln(x)

print("normalised after LN Class is ",normalised)