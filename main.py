import torch
from tokenlearner import TokenLearner, TokenLearnerModuleV11


tklr = TokenLearner(in_channels=128, num_tokens=8, use_sum_pooling=False)
tklr_v11 = TokenLearnerModuleV11(in_channels=128, num_tokens=8, num_groups=4, dropout_rate=0.)

x = torch.ones(256, 32, 32, 128)
y1 = tklr(x)
print(y1.shape)

tklr_v11.eval()
y2 = tklr_v11(x)
print(y2.shape)