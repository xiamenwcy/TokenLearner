import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLearner(nn.Module):
    """TokenLearner module.

    This is the module used for the experiments in the paper.

    Attributes:
      num_tokens: Number of tokens.
      use_sum_pooling: Whether to use the sum/average to aggregate the spatial feature after spatial attention
    """
    def __init__(self, in_channels, num_tokens, use_sum_pooling=False):
        """Applies learnable tokenization to the 2D inputs.

        Args:
          inputs: Inputs of shape `[bs, h, w, c]`.

        Returns:
          Output of shape `[bs, n_token, c]`.
        """
        super(TokenLearner, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.use_sum_pooling = use_sum_pooling
        self.norm = nn.LayerNorm(self.in_channels)  # Operates on the last axis (c) of the input data.

        self.attention_maps = nn.Sequential(
            # 3 layers of conv with gelu activation as suggested
            # in the paper.
            nn.Conv2d(self.in_channels, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            # This conv layer will generate the attention maps
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid()  # Note sigmoid for [0, 1] output
        )
        
    def forward(self, inputs):
        feature_shape = inputs.shape  # Shape:  [bs, h, w, c]

        selected = inputs
        selected = self.norm(selected)
        selected = selected.permute(0, 3, 1, 2)                  # Shape:  [bs, c, h, w]
        selected = self.attention_maps(selected)                 # Shape:  [bs, n_token, h, w]
        selected = selected.permute(0, 2, 3, 1)                  # Shape: [bs, h, w, n_token].
        selected = selected.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)  # Shape: [bs, h*w, n_token].
        selected = selected.permute(0, 2, 1)[..., None]   # Shape: [bs, n_token, h*w, 1].

         # Reshape the input to align it with the output of the conv block.
        feat = inputs
        feat = feat.view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)[:, None, ...]  # Shape: [bs, 1, h*w, c].

        # Element-Wise multiplication of the attention maps and the inputs
        attended_inputs = feat * selected  # (bs, n_token, h*w, c)

        if self.use_sum_pooling:
            outputs = torch.sum(attended_inputs, dim=2)   # (bs, n_token, c)
        else:
            outputs = torch.mean(attended_inputs, dim=2)

        return outputs


class TokenLearnerModuleV11(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.

    Instead of using 4 conv. layers with small channels to implement spatial
    attention, this version uses 2 grouped conv. layers with more channels. It
    also uses softmax instead of sigmoid. We confirmed that this version works
    better when having limited training data, such as training with ImageNet1K
    from scratch.

    Attributes:
      num_tokens: Number of tokens.
      dropout_rate: Dropout rate.
    """

    def __init__(self, in_channels, num_tokens, num_groups, dropout_rate=0.):
        """Applies learnable tokenization to the 2D inputs.

        Args:
          inputs: Inputs of shape `[bs, h, w, c]`.

        Returns:
          Output of shape `[bs, n_token, c]`.
        """
        super(TokenLearnerModuleV11, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.num_groups = num_groups  # in_channels and out_channels must both be divisible by groups
        self.norm = nn.LayerNorm(self.in_channels)  # Operates on the last axis (c) of the input data.

        self.attention_maps = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, groups=self.num_groups, bias=False),
            nn.Conv2d(self.in_channels, self.num_tokens, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
        )
        self.feat_conv = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, groups=self.num_groups, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        feature_shape = inputs.shape  # Shape:  [bs, h, w, c]

        selected = inputs
        selected = self.norm(selected)
        selected = selected.permute(0, 3, 1, 2)  # Shape:  [bs, c, h, w]
        selected = self.attention_maps(selected)  # Shape: [bs, n_token, h, w].
        selected = selected.permute(0, 2, 3, 1)  # Shape: [bs, h, w, n_token].
        selected = selected.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2],
                                 -1)  # Shape: [bs, h*w, n_token].
        selected = selected.permute(0, 2, 1)  # Shape: [bs, n_token, h*w].
        selected = F.softmax(selected, dim=-1)

        # Reshape the input to align it with the output of the conv block.
        feat = inputs
        feat = feat.permute(0, 3, 1, 2)   # Shape:  [bs, c, h, w]
        feat = self.feat_conv(feat)      # Shape: [bs, c, h, w].
        feat = feat.permute(0, 2, 3, 1)   # Shape: [bs, h, w, c].
        feat = feat.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)  # Shape: [bs, h*w, c].

        # Produced the attended inputs.
        outputs = torch.einsum("...si,...id->...sd",  selected, feat)  # (B, n_token, c)
        outputs = self.dropout(outputs)

        return outputs