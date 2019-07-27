import torch
import numpy as np

# abs
data1 = [-1, -2, 1, 2]
tensor1 = torch.FloatTensor(data1)        # 32bit

print(
    '\nabs',
    '\nnumpy:', np.sin(data1),        # [1 2 1 2]
    '\ntorch:', torch.sin(tensor1)       # [1 2 1 2]
)

data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)        # 32-bit floating point
print(
    '\nnumpy:', np.matmul(data, data),
    '\ntorch:', torch.mm(tensor, tensor)
)

