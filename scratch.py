import torch.nn as nn
import torch

input = torch.randn([3, 10, 1, 1])
print(input.shape)
m = nn.ConvTranspose2d(10, 6, kernel_size=10, padding=1, stride=2, bias=False)
out = m(input)
print(out.shape)