"""
Model definitions: ConvNeXt (LayerNorm, as in the paper), ConvNeXtBN (BatchNorm + head dropout), the SimpleCNN
baseline, the --model registry (MODELS) and build_model.

References:
    Block, ConvNeXt and LayerNorm are adapted from the official ConvNeXt implementation [2]
    (Copyright (c) Meta Platforms, Inc. and affiliates, MIT licence):
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    Z. Liu et al., "A ConvNet for the 2020s", CVPR 2022 [1]: https://arxiv.org/abs/2201.03545
    BatchNorm variant motivated by https://github.com/hunghuynh1110/Alzheimers-MRI-ConvNext
    ConvNeXt-Nano depths/dims from timm [21]: https://github.com/huggingface/pytorch-image-models

AI assistance: BlockBN, ConvNeXtBN, convnext_nano_bn and SimpleCNN were scaffolded with Claude (Anthropic), then
reviewed, trained and tested by me (see README, "Artificial Intelligence Usage Disclosure").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch (comment from the original authors [2])

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0. (each block's rate is set from dp_rates in ConvNeXt)
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # depthwise conv: groups=dim gives each channel its own 7x7 filter (no mixing between channels); padding=3 keeps H, W the same
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.norm = LayerNorm(dim, eps=1e-6) # the block's only norm (ResNet has one per conv)

        # inverted bottleneck: narrow -> wide (x4) ->
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # layer scale: one learnable value per channel, starts at 1e-6 so each block starts close to doing nothing
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # rate 0 -> Identity, so block is never skipped

    def forward(self, x):
        """(N, C, H, W) -> (N, C, H, W): the block keeps the shape and adds a learned correction to its input."""

        input = x # keep a copy for the residual connection

        x = self.dwconv(x) #depthwise convolution (looks at neigbouring values = spatial mixing)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C), as nn.Linear acts on the last dim

        x = self.norm(x) # channels last layer norm

        x = self.pwconv1(x) #the features at each pixel are combined (channel mixing), C -> 4C

        x = self.act(x) # gelu -> smooth version of ReLU, the activation used in Transformers. There's only one activation per block.

        x = self.pwconv2(x) # project back, 4C -> C

        if self.gamma is not None:
            x = self.gamma * x #scale layer per channel

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x) #residual: input + branch -> The block only has to learn a correction to its input

        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3 (all greyscale copies)
        num_classes (int): Number of classes. Default: 1000 (main.py passes 2: AD and NC)
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3] <- from tiny
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768] <- from tiny
        drop_path_rate (float): Stochastic depth rate of the last block. Default: 0.1
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_drop_rate (float): Dropout before the head. Default: 0 (as in the paper; --head_drop to change)
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.1,
                 layer_scale_init_value=1e-6, head_drop_rate=0.,
                 ): #param defaults

        super().__init__()

        self.downsample_layers = nn.ModuleList() # registered list of the stem + 3 downsampling layers, so PyTorch trains/saves their weights to this list
        
        # stem: "patchify" the image into non-overlapping 4x4 patches, (N, 3, 256, 256) -> (N, 96, 64, 64)
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), # stride 4 = jump 4 px, so 256/4 = 64; each patch -> 96 channels
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first") # normalise each pixel's 96 channels (dim 1) to mean 0, var 1; eps avoids /0
        )
        self.downsample_layers.append(stem)

        # 3 downsampling layers between stages: halve H, W and double channels, 96x64x64 -> 192x32x32 -> 384x16x16 -> 768x8x8
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"), # norm before the conv (unlike the stem) as stage outputs can drift; keeps training stable
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2), # non-overlapping 2x2 patches, (64 - 2)/2 + 1 = 32
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        
        # drop path rate per block: 18 evenly spaced values from 0 (first block) to drop_path_rate (last block)
        # deep blocks are skipped more often as they overfit most; .item() turns each 1-element tensor into a float
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0 # blocks built so far, so dp_rates[cur + j] gives each block its own rate
        for i in range(4): # 4 stages; every block is the same Block design, only the channels dims[i] changes (96, 192, 384, 768)
            stage = nn.Sequential( #chains the blocks so each one's output becomes the next one's input.
                # depths[i] = number of blocks in this stage (3, 3, 9, 3)
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            ) # blocks keep the shape, e.g. stage 1: 96x64x64 in -> 96x64x64 out
            self.stages.append(stage) #adds the stage to the ModuleList so its weights are registered
            cur += depths[i] # move past this stage's blocks so the next stage continues along dp_rates

        # final norm on the pooled (N, 768) vector
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # optional dropout on the pooled features (same place as ConvNeXtBN); 0 -> Identity, so the default is the paper's model
        self.head_drop = nn.Dropout(head_drop_rate) if head_drop_rate > 0 else nn.Identity()

        self.head = nn.Linear(dims[-1], num_classes) #classifier: a fully connected layer from 768 features to num_classes = 2 outputs, one score (logit) for AD and one for NC

        self.apply(self._init_weights) # runs _init_weights on every layer in the model (trained from scratch, so these are the starting weights)

    def _init_weights(self, m):
        """Initial weights for one layer (called on every layer via self.apply)."""

        # only Conv2d/Linear; LayerNorms keep weight 1, bias 0 and gamma stays 1e-6
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02) # small random weights break symmetry; truncated at +-2 std so none start too large
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """Stem + 4 stages + global average pooling + norm: (N, 3, 256, 256) -> pooled features (N, 768).
        Also used by visualise.py for the UMAP plots."""

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        """Images (N, 3, H, W) -> logits (N, num_classes); softmax of the logits gives the class probabilities."""

        x = self.forward_features(x)
        x = self.head_drop(x) # only active in model.train()
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        """Normalise each pixel's channels to mean 0 / variance 1, then apply the learnable per-channel scale and shift."""

        if self.data_format == "channels_last": # (N, H, W, C), used inside each Block
            # built-in only normalises the LAST dim(s), so it works directly when channels are last
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first": # (N, C, H, W), used in the stem and downsampling layers
            # built-in can't pick dim 1 (it errors), so normalise over the channels by hand
            u = x.mean(1, keepdim=True) # mean of the C channels at each pixel -> (N, 1, H, W)
            s = (x - u).pow(2).mean(1, keepdim=True) # variance at each pixel (divides by C, same as F.layer_norm)
            x = (x - u) / torch.sqrt(s + self.eps) # mean 0, var 1; eps avoids /0
            x = self.weight[:, None, None] * x + self.bias[:, None, None] # (C,) -> (C, 1, 1) so each channel gets its own learnable scale/shift
            return x


class BlockBN(nn.Module):
    r""" ConvNeXt Block with BatchNorm instead of LayerNorm, kept in (N, C, H, W) the whole way:
    DwConv -> BatchNorm2d -> 1x1 Conv -> GELU -> 1x1 Conv -> layer scale -> DropPath -> residual
    (implementation (1) from Block above, with BN in place of channels_first LN). Added for this project.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # same 7x7 depthwise conv as Block

        # BN normalises each channel over the batch and H, W (LN normalises over the channels at each pixel);
        # running mean/var are learnt during training and used at eval, so test images don't affect each other
        self.norm = nn.BatchNorm2d(dim)

        # 1x1 convs = the same per-pixel channel MLP as Block's nn.Linear, but without permuting to channels_last
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        # (C, 1, 1) so it broadcasts over H, W in channels_first
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """(N, C, H, W) -> (N, C, H, W); same steps as Block.forward, but with no permutes (channels_first throughout)."""
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = input + self.drop_path(x)
        return x

class ConvNeXtBN(nn.Module):
    r""" ConvNeXt with every LayerNorm swapped for BatchNorm, and dropout before the classifier.
        Alternative to ConvNeXt for training from scratch on the small ADNI dataset: the LN model sat at ~50%
        validation for the first ~16 epochs of convnext_t_scratch, and the BN-based reference repo
        (hunghuynh1110/Alzheimers-MRI-ConvNext) reports BN trained better for them.

    Args:
        same as ConvNeXt, plus
        head_drop_rate (float): Dropout on the pooled features before the head. Default: 0.4 (as in the reference repo)
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.1,
                 layer_scale_init_value=1e-6, head_drop_rate=0.4,
                 ):

        super().__init__()

        self.downsample_layers = nn.ModuleList()

        # stem: same 4x4 patchify, BN in place of LN, (N, 3, 256, 256) -> (N, 96, 64, 64)
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)

        # BN before each 2x2 stride-2 downsampling conv, as LN is in ConvNeXt
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.BatchNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[BlockBN(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # pooled (N, 768) -> BatchNorm1d -> Dropout -> Linear
        # dropout zeros a random 40% of the 768 features for each training image, so the head can't rely on a few of them
        self.norm = nn.BatchNorm1d(dims[-1])
        self.head_drop = nn.Dropout(head_drop_rate)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initial weights for one layer (called on every layer via self.apply)."""
        # same as ConvNeXt; BatchNorms keep PyTorch's defaults (weight 1, bias 0)
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """Stem + 4 stages + global average pooling + BatchNorm1d: (N, 3, 256, 256) -> pooled features (N, 768)."""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        """Images (N, 3, H, W) -> logits (N, num_classes)."""
        x = self.forward_features(x)
        x = self.head_drop(x) # only active in model.train(); model.eval() turns it off
        x = self.head(x)
        return x


class SimpleCNN(nn.Module):
    r""" Baseline for the open research dilemma: a plain CNN with none of ConvNeXt's design choices
    (no patchify stem, no large-kernel depthwise convs, no inverted bottleneck, no LayerNorm/layer scale/drop path,
    no residual connections).

    4 blocks of 3x3 conv -> BatchNorm2d -> ReLU -> 2x2 max-pool, (N, 3, 256, 256) -> (N, 256, 16, 16),
    then global average pooling -> dropout -> linear head. About 0.39 M parameters (ConvNeXt-Nano: 15.0 M).
    Trained with exactly the same data, input (incl. 2.5D), augmentation, optimiser and schedule as ConvNeXt,
    so only the architecture differs.

    Args:
        in_chans (int): Number of input channels. Default: 3
        num_classes (int): Number of classes. Default: 2
        dims (tuple(int)): Channels of the 4 blocks. Default: (32, 64, 128, 256)
        head_drop_rate (float): Dropout before the head. Default: 0.4 (same as the BN ConvNeXt head)
    """
    def __init__(self, in_chans=3, num_classes=2, dims=(32, 64, 128, 256), head_drop_rate=0.4, **unused):
        super().__init__()
        # main.py passes ConvNeXt-only options (drop_path_rate, layer_scale_init_value); they don't apply here
        layers, c_in = [], in_chans
        for c_out in dims:
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False), # bias is redundant before BN
                       nn.BatchNorm2d(c_out), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head_drop = nn.Dropout(head_drop_rate)
        self.head = nn.Linear(dims[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # standard init for ReLU nets
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """4 conv blocks + global average pooling: (N, 3, 256, 256) -> pooled features (N, 256)."""
        return self.features(x).mean([-2, -1]) # global average pooling, (N, 256, 16, 16) -> (N, 256)

    def forward(self, x):
        """Images (N, 3, H, W) -> logits (N, num_classes)."""
        return self.head(self.head_drop(self.forward_features(x)))


# ---------------------------------------------------------------------------
# Model sizes (--model)
# ---------------------------------------------------------------------------

def convnext_tiny(**kwargs): #kwargs is the dictionary of initial arguments
    """ConvNeXt-T as in the paper (LayerNorm), 27.8 M parameters."""
    # depths = how many ConvNeXt blocks each of the 4 stages has, dims = how many channels (feature maps) each stage has
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

def convnext_tiny_bn(**kwargs):
    """ConvNeXt-T with BatchNorm and head dropout (the final model), 27.8 M parameters."""
    return ConvNeXtBN(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs) # same size as convnext_tiny, BN + head dropout

def convnext_nano_bn(**kwargs):
    """Smaller BatchNorm ConvNeXt (Nano), 15.0 M parameters, for the cost/benefit comparison."""
    # not in the ConvNeXt paper (it starts at Tiny): Nano's depths/dims from timm (R. Wightman, pytorch-image-models),
    # same blocks as Tiny, fewer and narrower: 15.0 M parameters vs 27.8 M
    return ConvNeXtBN(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)

def simple_cnn(**kwargs):
    """The baseline CNN for the open research dilemma, 0.39 M parameters."""
    return SimpleCNN(**kwargs)

# --model name -> factory function, e.g. MODELS['convnext_tiny_bn'](num_classes=2)
MODELS ={f.__name__: f for f in (convnext_tiny, convnext_tiny_bn, convnext_nano_bn, simple_cnn)}


def build_model(args):
    """Build --model from the training arguments (also used to rebuild a trained model from a checkpoint's saved args).
    Every model is trained from scratch (random initial weights) and has 2 outputs: AD (0) and NC (1)."""
    if args.model not in MODELS:
        raise ValueError(f"Unknown model '{args.model}'. Available: {', '.join(MODELS)}")
    kwargs = dict(
        num_classes=2,
        drop_path_rate=args.drop_path, # reduce overfitting by dropping blocks in the model
        layer_scale_init_value=args.layer_scale_init_value, # per-channel scale on each block's output
    )
    # None = each model's own default (0.4 for the BN models and simple_cnn, 0 for LayerNorm ConvNeXt);
    # getattr: checkpoints saved before --head_drop existed don't have it
    if getattr(args, 'head_drop', None) is not None:
        kwargs['head_drop_rate'] = args.head_drop
    return MODELS[args.model](**kwargs)

