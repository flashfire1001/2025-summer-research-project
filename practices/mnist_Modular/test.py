import torch

t = torch.tensor([[1,2,3,4,5],[6,7,8,9,0]],dtype=torch.int) # shape(2,5)

print(t.view(5,2))