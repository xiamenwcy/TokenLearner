# TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ariG23498/TokenLearner/blob/master/TokenLearner.ipynb) 

<div align="center">
  <img src="https://blogger.googleusercontent.com/img/a/AVvXsEiylT3_nmd9-tzTnz3g3Vb4eTn-L5sOwtGJOad6t2we7FsjXSpbLDpuPrlInAhtE5hGCA_PfYTJtrIOKfLYLYGcYXVh1Ksfh_C1ZC-C8gw6GKtvrQesKoMrEA_LU_Gd5srl5-3iZDgJc1iyCELoXtfuIXKJ2ADDHOBaUjhU8lXTVdr2E7bCVaFgVHHkmA=s1600"><br>
  <small>Source: <a href="https://ai.googleblog.com/2021/12/improving-vision-transformer-efficiency.html">Improving Vision Transformer Efficiency and Accuracy by Learning to Tokenize</a></small>
</div><br>

A PyTorch implementation of TokenLearner: What Can 8 Learned Tokens Do for Images and Videos? [1-2].
Unlike another Unofficial PyTorch implementation [3], our version is heavily borrowed from the official implementation [4] and TensorFlow implementation[5],
and try to keep consistent with them.

## Usage
You can access the `TokenLearner` and `TokenLearnerModuleV11` class from the `tokenlearner` file. You can use this layer with a Vision Transformer, MLPMixer, or Video Vision Transformer as done in the paper.

```python
import torch
from tokenlearner import TokenLearner

tklr = TokenLearner(in_channels=128, num_tokens=8, use_sum_pooling=False)

x = torch.ones(256, 32, 32, 128)
y1 = tklr(x)
print(y1.shape)  # [256, 8, 128]
```

You can also use `TokenLearnerModuleV11`, which aligns with the [official implementation](https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py).

```python
import torch
import torch.nn as nn
from tokenlearner import TokenLearnerModuleV11

tklr_v11 = TokenLearnerModuleV11(in_channels=128, num_tokens=8, num_groups=4, dropout_rate=0.)

tklr_v11.eval()  # control droput
x = torch.ones(256, 32, 32, 128)
y2 = tklr_v11(x)
print(y2.shape)  # [256, 8, 128]
```


# References

[1] TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?; Ryoo et al.; arXiv 2021; https://arxiv.org/abs/2106.11297

[2] TokenLearner: Adaptive Space-Time Tokenization for Videos; Ryoo et al., NeurIPS 2021; https://openreview.net/forum?id=z-l1kpDXs88

[3] [Unofficial PyTorch implementation](https://github.com/rish-16/tokenlearner-pytorch)

[4] [official implementation](https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py)

[5] [TensorFlow implementation](https://github.com/ariG23498/TokenLearner)