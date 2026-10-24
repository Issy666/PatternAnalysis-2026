# ConvNeXt for Alzheimer's Disease Classification (ADNI)

## Description of Algorithm

### Step 1: Patchify Stem (Downsampling)

Once the slices have been loaded and normalised, the first layer of the network is the **stem**. It turns raw pixels into feature maps (`modules.py`, `downsample_layers[0]`):

```python
stem = nn.Sequential(
    nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
)
```

**Input.** ADNI slices are 256×240 px greyscale JPEGs. `PadToSquare` adds 8 black rows at the top and bottom to make them 256×256 without stretching the brain. `ImageFolder` loads each slice as RGB, which copies the grey channel into 3 identical channels to match the stem's default `in_chans=3`. The input to the stem is therefore `(N, 3, 256, 256)`, where `N` is the batch size.

**Patchify convolution.** The 4×4 kernel with stride 4 moves 4 pixels at a time, so the patches don't overlap. The image is cut into a grid of 4×4 patches, and each patch becomes one 96-channel feature vector:

```
out = (in − kernel + 2·padding) / stride + 1 = (256 − 4 + 0) / 4 + 1 = 64
(N, 3, 256, 256)  →  (N, 96, 64, 64)
```

**LayerNorm (channels first).** At every pixel position of every image, the 96 channel values are normalised to mean 0 and variance 1, then scaled and shifted by learnable per-channel weights. Each position uses only its own statistics, so the result doesn't depend on the batch size or the other images in the batch (unlike BatchNorm). `channels_first` normalises over dimension 1 of the `(N, C, H, W)` tensor that `Conv2d` outputs. PyTorch's built-in `nn.LayerNorm` only normalises the last dimension, which is why `modules.py` defines its own `LayerNorm`. `eps=1e-6` stops a division by zero on flat regions such as the black background. The shape is unchanged: `(N, 96, 64, 64)`.

**Why these numbers.**

| Setting | Value | Reason |
|---|---|---|
| Kernel size and stride | 4 and 4 | Taken from Swin Transformer's 4×4 patches. The paper's ResNet-50 stem (7×7 conv stride 2 + max pool stride 2) also shrinks the image 4×, so swapping it for one simpler non-overlapping conv keeps the same resolution and slightly improved ImageNet accuracy (79.4% → 79.5%) [1]. ViT's 16×16 patches would leave only 16×16 positions, too coarse for small structures such as the hippocampus. |
| Output channels `dims[0]` | 96 | The channel width of ConvNeXt-T, matching Swin-T [1]. Each patch holds 16 grey pixels (48 values after the 3-channel copy), so 96 channels are enough to represent a patch without losing information. |
| LayerNorm after the stem | `eps=1e-6` | The paper adds a LayerNorm wherever the spatial resolution changes (after the stem and before each downsampling layer) to keep training stable [1]. |
| Input size | 256 | ADNI's native width, so no resizing is needed after padding. 256 is divisible by 32 (the stem's 4× times three later 2× downsamples), so every stage divides evenly: 64 → 32 → 16 → 8. The paper used 224 for ImageNet. |

**Why the stem is kept as in the paper for ADNI.**
- The 8 padding rows are exactly 2 patch rows, so no patch mixes padding with brain tissue.
- 96 channels are more than enough to hold each 4×4 greyscale patch, so the 4× downsampling loses no information at this step.
- `in_chans=1` would remove the copied channels. It isn't used because the network is trained from scratch, so three identical channels behave the same as one (the stem's weights for each channel simply add together). It would also add very little compute and would require changing timm's RGB augmentation pipeline and the cached normalisation statistics.
- Keeping the paper's stem keeps the comparison with the ConvNeXt-T benchmark [1] direct.

After the stem, the `(N, 96, 64, 64)` feature maps go into Stage 1: 3 ConvNeXt blocks, which keep the shape the same.

### Stages

The network has 4 **stages**. Each stage is a chain of ConvNeXt blocks working at one resolution and channel width (`modules.py`):

```python
for i in range(4):
    stage = nn.Sequential(
        *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
    )
    self.stages.append(stage)
    cur += depths[i]
```

- The list comprehension builds `depths[i]` blocks for stage `i`. Every block uses the same `Block` design. Only its width `dims[i]` differs between stages.
- `*` unpacks the list, because `nn.Sequential` takes layers as separate arguments (`nn.Sequential(b1, b2, b3)`), not as one list.
- `nn.Sequential` chains the blocks, so each block's output is the next block's input.
- `self.stages` is an `nn.ModuleList`, so PyTorch registers every block's weights for training, `.cuda()` and checkpoints.
- Each block gets its own drop path rate `dp_rates[cur + j]`. `cur` counts the blocks already built, so the rates continue from one stage to the next (see [Drop path](#drop-path-stochastic-depth)).

Blocks keep the input shape, so only the downsampling layers between stages change the resolution:

| Stage | Blocks | Channels | `dp_rates` indices | Shape (256 input) |
|---|---|---|---|---|
| 1 | 3 | 96 | 0–2 | 96 × 64 × 64 |
| 2 | 3 | 192 | 3–5 | 192 × 32 × 32 |
| 3 | 9 | 384 | 6–14 | 384 × 16 × 16 |
| 4 | 3 | 768 | 15–17 | 768 × 8 × 8 |

**Why 3, 3, 9, 3.** ResNet-50 uses (3, 4, 6, 3) blocks per stage. ConvNeXt changed this to Swin-T's 1:1:3:1 ratio, which puts most of the compute in Stage 3 and improved ImageNet accuracy [1]. At 16 × 16, Stage 3 keeps enough spatial detail while each block is still cheap. For ADNI, each Stage 3 position covers 16 × 16 px of the input slice. Its receptive field, the region of the input it can see, is much larger, about the scale of the hippocampus and ventricles.

**Kept as in the paper for now.** The stage loop works for any `depths` and `dims`, so smaller variants only need new model functions next to `convnext_tiny`. The first from-scratch run (27.8 M parameters) showed signs of overfitting: training accuracy kept rising to 85% while validation stalled at around 73–78%. Smaller ConvNeXt models are listed as a future experiment in `future_experiments.txt`.

### ConvNeXt Block

Every stage is built from the same `Block` (`modules.py`). Each block takes `(N, C, H, W)` and returns the same shape. Shapes below are for Stage 1 (C = 96, 64 × 64):

```python
def forward(self, x):
    input = x
    x = self.dwconv(x)               # 7×7 depthwise conv          (N, 96, 64, 64)
    x = x.permute(0, 2, 3, 1)        # channels last               (N, 64, 64, 96)
    x = self.norm(x)                 # LayerNorm                   (N, 64, 64, 96)
    x = self.pwconv1(x)              # Linear 96 → 384             (N, 64, 64, 384)
    x = self.act(x)                  # GELU                        (N, 64, 64, 384)
    x = self.pwconv2(x)              # Linear 384 → 96             (N, 64, 64, 96)
    if self.gamma is not None:
        x = self.gamma * x           # layer scale, per channel
    x = x.permute(0, 3, 1, 2)        # back to channels first      (N, 96, 64, 64)
    x = input + self.drop_path(x)    # residual connection
    return x
```

A block does two jobs, the same split as a Transformer block: first it mixes information across **positions** (the depthwise conv, in place of attention), then it mixes information across **features** (the MLP).

**1. Depthwise 7×7 convolution (spatial mixing).** `nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)`.
- `groups=dim` makes it *depthwise*: each channel has its own 7×7 filter and only looks at itself, with no mixing between channels. This needs 96 × 7 × 7 weights instead of 96 × 96 × 7 × 7 for a normal conv.
- `kernel_size=7` lets each pixel see a 7 × 7 neighbourhood. ResNet uses 3 × 3. The paper raised it to 7, matching Swin's 7 × 7 attention window, and found no further gain from larger kernels [1].
- `padding=3` keeps the size the same: `(64 − 7 + 2·3) / 1 + 1 = 64`.
- On ADNI, a 7 × 7 window covers 28 × 28 px of the input slice at Stage 1, and 112 × 112 px at Stage 3, a large part of the brain.

**2. Permute to channels last.** `nn.Linear` acts on the last dimension, so the channels are moved to the end. A `1×1 Conv2d` on channels-first data is mathematically the same, but the authors found `Linear` on channels last slightly faster in PyTorch (block docstring). This is why the custom LayerNorm also has a `channels_last` mode.

**3. LayerNorm.** This is the block's only normalisation: each pixel's channels are normalised (see Custom LayerNorm). ResNet has a BatchNorm after every conv. ConvNeXt uses one LayerNorm per block, as Transformers do, and the paper found that fewer norms trained slightly better [1].

**4. Inverted bottleneck: Linear → GELU → Linear (channel mixing).**
- This is a small MLP applied to each pixel separately, with the same weights at every position. This is where the channels are combined, which the depthwise conv does not do.
- *Inverted bottleneck:* ResNet's bottleneck goes wide → narrow → wide. ConvNeXt goes narrow → wide (×4) → narrow (96 → 384 → 96), like a Transformer MLP and MobileNetV2. The wide middle layer gives room to learn combinations of features.
- *GELU* is a smooth version of ReLU, the activation used in Transformers. There is only one activation per block, where ResNet has one after every conv, and the paper found that fewer activations improved accuracy [1].
- These two layers hold most of the block's weights: about 74 k of the 79 k parameters in a Stage 1 block (dwconv 4,800 + norm 192 + pwconv1 37,248 + pwconv2 36,960 + gamma 96 = 79,296).

**5. Layer scale (`gamma`).** The branch's output is multiplied by one learnable value per channel, starting at 1e-6 (see [Layer scale](#layer-scale)). If `--layer_scale_init_value` is 0 or less, `gamma` is `None` and this step is skipped.

**6. Residual connection and drop path.** `x = input + self.drop_path(x)` adds the block's original input back, so the block only has to learn a *correction* to its input. Gradients can also flow straight back through the `+` across all 18 blocks, which makes the deep network trainable. During training, `drop_path` sometimes zeros the branch so the block is skipped for that image (see [Drop path](#drop-path-stochastic-depth)). With a rate of 0, it is `nn.Identity()` and does nothing.

**Why it is kept as in the paper for ADNI.** No part of the block depends on the dataset: every size comes from `dim`, and it works on whatever features the stem produces.
- The 7 × 7 kernel already covers structures such as the ventricles and hippocampus by Stage 3.
- LayerNorm, GELU and layer scale are there for training stability rather than for any property of the data.
- The depthwise conv keeps the CNN's local, translation-equivariant inductive bias, which suits a small dataset like ADNI better than attention.

Two block-related ideas are listed in `future_experiments.txt`:
- A higher drop path rate (0.2–0.3), to reduce the overfitting seen in the first run.
- Coordinate attention, as in CA-ConvNeXt [7]. Global average pooling discards *where* a feature was, but in brain MRI location matters (e.g. atrophy in the medial temporal lobe). This should only be tried once the overfitting is under control.

### Downsampling Between Stages

The other 3 entries of `downsample_layers` sit between the 4 stages. Each one runs after a stage finishes (`modules.py`):

```python
for i in range(3):
    downsample_layer = nn.Sequential(
        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
    )
```

The loop runs for `i` = 0, 1, 2. Each layer takes `dims[i]` channels to `dims[i+1]` channels (`dims` has 4 entries, so only 3 transitions are possible):

| Layer | Runs after | Channels | Output shape (256 input) |
|---|---|---|---|
| Stem | input image | 3 → 96 | 96 × 64 × 64 |
| `downsample_layers[1]` | Stage 1 (3 blocks) | 96 → 192 | 192 × 32 × 32 |
| `downsample_layers[2]` | Stage 2 (3 blocks) | 192 → 384 | 384 × 16 × 16 |
| `downsample_layers[3]` | Stage 3 (9 blocks) | 384 → 768 | 768 × 8 × 8 |

Stage 4 (3 blocks) then works on the 768 × 8 × 8 maps before global average pooling and the classifier head.

**LayerNorm, then convolution.** The LayerNorm normalises each pixel's channels exactly as in the stem. Here it comes **before** the conv because its input is the output of a stage of residual blocks, whose values can drift as blocks add to them. The stem's input is already normalised by the data pipeline, so the stem normalises after its conv. The paper found that training became unstable with separate downsampling layers unless a LayerNorm was added wherever the resolution changes [1].

**2×2 convolution, stride 2.** This is a smaller non-overlapping patchify step. It halves the height and width, `(64 − 2) / 2 + 1 = 32`, and doubles the channels. ResNet downsamples inside its blocks with a stride-2 3×3 conv. ConvNeXt follows Swin Transformer and uses a separate 2×2 stride-2 layer, which improved ImageNet accuracy in the paper [1].

**Why halve the size and double the channels.** Each halving of height and width leaves 4× fewer positions, so doubling the channels keeps the compute per stage roughly balanced. This builds a feature pyramid: the network trades spatial detail (*where* something is) for feature detail (*what* it is). For ADNI, Stage 1 at 64 × 64 sees fine edges and tissue boundaries. Stages 3 and 4 at 16 × 16 and 8 × 8 have each position covering a large part of the slice (32 × 32 px of the input at Stage 4), so they can capture whole structures such as enlarged ventricles and atrophy in the medial temporal lobe.

**Why it is kept as in the paper for ADNI.** With the 256 input, every halving divides evenly (64 → 32 → 16 → 8), so no rows or columns are lost. The final 8 × 8 map is still fine enough for Grad-CAM to show which region of the brain the model used. The paper's ImageNet setting at 224 gives 7 × 7.

### Pooling, Final Norm and Classifier Head

After Stage 4, the features are turned into a single prediction per image (`modules.py`):

```python
self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
self.head = nn.Linear(dims[-1], num_classes)
```

| Step | Code | Shape |
|---|---|---|
| Stage 4 output | | `(N, 768, 8, 8)` |
| Global average pooling | `x.mean([-2, -1])` in `forward_features` | `(N, 768)` |
| Final LayerNorm | `self.norm` | `(N, 768)` |
| Classifier head | `self.head` | `(N, 2)` |

**Global average pooling** averages each of the 768 channels over all 8 × 8 positions, giving one feature vector per image. Each channel then says how strongly its feature appears anywhere in the slice. The result doesn't depend on exactly where in the slice a feature appears, which helps with differences in head position and shape between patients. It also has no parameters, unlike flattening the 8 × 8 × 768 map into a large fully connected layer.

**Final LayerNorm.** `dims[-1]` is the last width, 768. The pooled vector is normalised to mean 0 and variance 1, plus a learnable scale and shift, so the classifier receives values in a stable range. This uses PyTorch's built-in `nn.LayerNorm` instead of the custom class: after pooling, the channels are the last dimension, which is what the built-in version normalises.

**Classifier head.** A fully connected layer maps the 768 features to `num_classes` = 2 scores (logits), one for AD and one for NC. During training, cross-entropy with label smoothing is applied to these logits. At test time, softmax turns them into probabilities, and that probability is the confidence score used for the reliability diagram and the reject threshold in the open research dilemma.

### Weight Initialisation

```python
self.apply(self._init_weights)

def _init_weights(self, m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        trunc_normal_(m.weight, std=.02)
        nn.init.constant_(m.bias, 0)
```

`self.apply(fn)` runs `fn` on every layer in the model, recursively: the stem, the downsampling layers, all 18 blocks and the head. Only `Conv2d` and `Linear` layers are changed:
- **Weights** start as small random values from a normal distribution with standard deviation 0.02. Random values break symmetry: if every weight started equal, every channel would learn the same feature. Small values stop the outputs from growing too large through 18 residual blocks.
- **Truncated** means any value beyond ±2 standard deviations is redrawn, so no weight starts unusually large.
- **Biases** start at 0.

LayerNorms keep PyTorch's defaults (weight 1, bias 0), and each block's layer scale `gamma` stays at `--layer_scale_init_value` (1e-6). This is the same initialisation as ViT and Swin, and the paper uses it too [1]. It matters here because the model is trained from scratch: with no ImageNet weights, these random values are its only starting point.

### Custom LayerNorm (Channels First and Channels Last)

ConvNeXt uses LayerNorm on two tensor layouts, so `modules.py` defines its own `LayerNorm` class that supports both:

| Mode | Tensor shape | Used in | Channels are at |
|---|---|---|---|
| `channels_last` (default) | `(N, H, W, C)` | Inside each `Block`, after the permute for the `nn.Linear` layers | last dim |
| `channels_first` | `(N, C, H, W)` | Stem and downsampling layers, directly after or before a `Conv2d` | dim 1 |

**Why a custom class is needed.** PyTorch's built-in `F.layer_norm` has no option to choose the dimension. It always normalises over the **last** dimension(s), and `normalized_shape` must match them. On a conv output of shape `(2, 96, 64, 64)`, asking it to normalise the 96 channels fails:

```
F.layer_norm(x, (96,))
RuntimeError: Given normalized_shape=[96], expected input with shape [*, 96], but got input of size[2, 96, 64, 64]
```

Passing `(96, 64, 64)` runs, but it normalises C, H and W together, giving one mean per image instead of one per pixel, which is a different operation. The alternative is to permute to channels last, normalise and permute back, which costs 2 extra reshuffles of the tensor every time. The ConvNeXt authors instead compute the statistics over dim 1 directly:

```python
def forward(self, x):
    if self.data_format == "channels_last":
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    elif self.data_format == "channels_first":
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
```

- **`channels_last`** passes straight to the fast built-in function.
- **`u` and `s`** are the mean and variance of the C channel values at each pixel. `keepdim=True` keeps shape `(N, 1, H, W)`, so one mean and variance per pixel broadcasts across all C channels.
- **`s`** divides by C (not C − 1), the same definition `F.layer_norm` uses, so both modes give the same result. `eps` stops a division by zero where all channels are equal, such as on the black background.
- **`self.weight[:, None, None]`** reshapes the per-channel scale from `(C,)` to `(C, 1, 1)`. Without that, PyTorch would match `(C,)` against the last dimension (W) and scale the wrong axis. The same applies to the bias.

For example, a pixel with channel values `[2, 4, 6]` has mean 4 and standard deviation 1.63, so it becomes `[−1.22, 0, 1.22]`. A pixel with `[100, 200, 300]` gives the same result. The pattern across channels is kept, but the overall scale is removed. Each pixel is normalised using only its own channels, so unlike BatchNorm the result doesn't depend on the batch size or on the other images in the batch.

## Problem

## How it Works
1. Run `main.py` from this directory:

   ```bash
   python3 main.py
   ```

   Any of the variables below can be overridden on the command line, e.g.

   ```bash
   python3 main.py --batch_size 32 --epochs 100 --data_path /path/to/AD_NC/train
   ```

   These defaults have been fine-tuned to give the best performing model for classifying MRI scans as AD (Alzheimer's disease) or NC (normal control). Boolean flags take a value, e.g. `--model_ema True`.

### Training

| Argument | Default | Purpose |
|---|---|---|
| `--batch_size` | `64` | Number of images per batch (per GPU). |
| `--epochs` | `150` | Number of passes over the training set. |
| `--update_freq` | `1` | Gradient accumulation steps; the effective batch size is `batch_size × update_freq`. |
| `--start_epoch` | `0` | Epoch to start from (used when resuming training). |
| `--seed` | `0` | Random seed for reproducibility. |
| `--device` | `cuda` | Device to train/evaluate on (`cuda` or `cpu`). |
| `--num_workers` | `4` | Number of DataLoader worker processes for loading images. |
| `--pin_mem` | `True` | Pin CPU memory in the DataLoader for faster transfer to the GPU. |
| `--use_amp` | `True` | Automatic mixed precision (float16 where safe); much faster on an RTX GPU. |

### Model

| Argument | Default | Purpose |
|---|---|---|
| `--model` | `convnext_tiny` | Name of the model architecture to train (a model function in `modules.py`). |
| `--input_size` | `256` | Height and width (pixels) images are resized to before entering the network. |
| `--drop_path` | `0.1` | Stochastic depth rate (randomly drops residual blocks during training for regularisation). See below. |
| `--layer_scale_init_value` | `1e-6` | Initial value of ConvNeXt's per-channel layer scale parameters. See below. |
| `--nb_classes` | `2` | Number of output classes (AD and NC). |

#### Drop path (stochastic depth)

Each ConvNeXt block has two paths: a **residual branch** (the block's convolutions and layers, which learn features) and a **shortcut** that carries the input forward unchanged. Their outputs are added, `output = input + branch(input)`, so each block learns a correction on top of its input.

During training, drop path randomly discards the residual branch for individual images, so `output = input` and that block is skipped for that image. The skip probability rises linearly with depth, from 0 for the first block to `--drop_path` for the last. Early blocks learn basic features such as edges and are rarely skipped. Deep blocks are skipped most often. At test time every block always runs.

The per-block rates are set in `modules.py`:

```python
dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
```

`sum(depths)` = 3 + 3 + 9 + 3 = 18 blocks, so `torch.linspace` gives 18 evenly spaced rates from 0 to `--drop_path`, including both ends (a step of 0.1 / 17 ≈ 0.0059). `.item()` turns each one from a 1-element tensor into a Python float. A counter `cur` tracks how many blocks have already been built, so `dp_rates[cur + j]` gives each block its own rate. A rate of 0 makes `DropPath` an `nn.Identity()`, so the first block is never skipped.

| Stage | Blocks | Drop path rates (`--drop_path 0.1`) |
|---|---|---|
| 1 | 1–3 | 0.000 – 0.012 |
| 2 | 4–6 | 0.018 – 0.029 |
| 3 | 7–15 | 0.035 – 0.082 |
| 4 | 16–18 | 0.088 – 0.100 |

Because the network can't rely on any single deep block, it overfits less. This matters for ADNI: the training slices come from only about a thousand patients (20 slices each), and neighbouring slices are very similar. `0.1` is the value the ConvNeXt paper uses for ConvNeXt-T [1]. Try `0.2`–`0.3` if validation accuracy falls well behind training accuracy.

#### Layer scale

Layer scale multiplies each block's residual branch by a learnable vector `gamma`, with one value per channel: `output = input + gamma * branch(input)`. Every value in `gamma` starts at `--layer_scale_init_value` and is learned during training like any other weight.

Starting at `1e-6` means each branch adds almost nothing at first, so the 18-block network begins close to an identity mapping (`output ≈ input`). Each block then "switches on" gradually as its `gamma` grows. This keeps early training stable: without it, the randomly initialised branches of all 18 blocks add noise together from the first step, which can cause large, unstable gradients at a high learning rate such as `4e-3`. The per-channel values also let the network learn how much each feature in each block should contribute.

`1e-6` is the value from the ConvNeXt paper [1].

### Exponential Moving Average (EMA)

| Argument | Default | Purpose |
|---|---|---|
| `--model_ema` | `False` | Keep an exponential moving average of the model weights. |
| `--model_ema_decay` | `0.9999` | Decay rate of the EMA weights. |
| `--model_ema_force_cpu` | `False` | Store the EMA model on the CPU to save GPU memory. |
| `--model_ema_eval` | `False` | Also evaluate the EMA model on the validation set during training. |

### Optimisation

| Argument | Default | Purpose |
|---|---|---|
| `--opt` | `adamw` | Optimiser. |
| `--opt_eps` | `1e-8` | Optimiser epsilon (numerical stability term). |
| `--opt_betas` | `None` | Optimiser betas; `None` uses the optimiser's defaults. |
| `--momentum` | `0.9` | Momentum (only used by the SGD optimiser). |
| `--clip_grad` | `None` | Maximum gradient norm; `None` disables gradient clipping. |
| `--lr` | `4e-3` | Peak learning rate. |
| `--min_lr` | `1e-6` | Lowest learning rate reached by the cosine schedule. |
| `--warmup_epochs` | `20` | Number of epochs the learning rate linearly warms up to `--lr`. |
| `--warmup_steps` | `-1` | Number of warm-up steps; overrides `--warmup_epochs` if greater than 0. |
| `--weight_decay` | `0.05` | Weight decay (L2 regularisation strength). |
| `--weight_decay_end` | `None` | Final weight decay for the cosine weight decay schedule; `None` keeps it constant. |

### Augmentation and Regularisation

| Argument | Default | Purpose |
|---|---|---|
| `--aa` | `''` (off) | AutoAugment/RandAugment policy, e.g. `rand-m9-mstd0.5-inc1`. |
| `--color_jitter` | `0` (off) | Strength of random brightness/contrast/saturation changes. |
| `--train_interpolation` | `bilinear` | Interpolation used when resizing training images (`random`, `bilinear`, `bicubic`). |
| `--smoothing` | `0.1` | Label smoothing applied to the loss. |
| `--reprob` | `0.25` | Probability of applying random erasing to a training image. |
| `--remode` | `pixel` | Random erasing mode (what the erased region is filled with). |
| `--recount` | `1` | Number of regions erased per image. |
| `--crop_pct` | `None` | Centre crop ratio used at evaluation time. |

### Mixup and CutMix

| Argument | Default | Purpose |
|---|---|---|
| `--mixup` | `0.8` | Mixup alpha; mixup is enabled if > 0. |
| `--cutmix` | `1.0` | CutMix alpha; CutMix is enabled if > 0. |
| `--cutmix_minmax` | `None` | CutMix min/max box ratio; overrides `--cutmix` alpha if set. |
| `--mixup_prob` | `1.0` | Probability of applying mixup or CutMix to a batch. |
| `--mixup_switch_prob` | `0.5` | Probability of using CutMix instead of mixup when both are enabled. |
| `--mixup_mode` | `batch` | Granularity of mixup/CutMix parameters: per `batch`, `pair` or `elem`. |

### Data

| Argument | Default | Purpose |
|---|---|---|
| `--data_set` | `ADNI` | Dataset to use (`ADNI` or `CIFAR`). |
| `--data_path` | `/home/isabel/Desktop/data/ADNI/AD_NC/train` | Training images (folder containing `AD/` and `NC/`). |
| `--val_data_path` | `/home/isabel/Desktop/data/ADNI/AD_NC/validation` | Validation images, used for checkpoint selection and tuning. |
| `--eval_data_path` | `/home/isabel/Desktop/data/ADNI/AD_NC/test` | Held-out test images, used for the final reported result. |
| `--stats_dir` | `stats/` (next to `normalise.py`) | Where the training-set mean/std used for normalisation is cached. |

### Checkpoints, Evaluation and Logging

| Argument | Default | Purpose |
|---|---|---|
| `--output_dir` | `/home/isabel/Desktop/results` | Where checkpoints and logs are saved; empty for no saving. |
| `--log_dir` | `/home/isabel/Desktop/log` | Where TensorBoard logs are written. |
| `--save_ckpt` | `True` | Save checkpoints to `--output_dir` (`checkpoint-best.pth` is the best validation epoch). |
| `--save_ckpt_freq` | `1` | Save an epoch checkpoint every N epochs. |
| `--save_ckpt_num` | `3` | Number of recent epoch checkpoints to keep. |
| `--resume` | `''` | Checkpoint to resume training from, or to evaluate with `--eval True`. |
| `--auto_resume` | `True` | Resume from the latest checkpoint in `--output_dir` if one exists. |
| `--eval` | `False` | Only evaluate a trained model on the test set (no training). |
| `--disable_eval` | `False` | Skip validation during training. |

-----------------------------------------------------

## Feasibility Review
### User Need
The intended users of this Alzheimer's detection tool include:
- Doctors and practitioners, to assist in medical diagnosis (as a screening aid).
- Researchers studying Alzheimer's disease, its features, and prediction/detection accuracy.
- Researchers studying machine learning and its application to medical datasets and image processing.

### User Scope
The purpose of this project is to develop a ConvNeXt model as a prototype machine learning computer vision model that classifies 2D T1-weighted brain MRI slices as Alzheimer's Disease (AD) or Cognitively Normal (CN). The model should flag ambiguous or low-confidence classifications for review by a human specialist. Ambiguous cases should be reliably identified, and the predicted confidence should correlate with diagnostic correctness. This is the open research dilemma for Project 2.1: does the model know when it is unsure, or does it stay overconfident when it is wrong?

### Acceptance Criteria
The minimum acceptance criteria are as follows:
- Minimum classification test accuracy of 80% for AD versus CN.
  *Test:* accuracy of the ADNI test split.
- A bias towards false positives over false negatives for misclassifications (missing AD is worse than a false alarm).
  *Test:* AD recall ≥ CN recall, from the confusion matrix (records TP, TN, FP, FN for test data), along with per-class precision, recall and macro-F1 (accuracy for each class).
- Can identify ambiguous vs confident classifications, where confidence correlates with diagnostic correctness.
  *Test:* reliability diagram (test predictions sorted via confidence bins for each classification and plot the actual accuracy for the corrosponding confidence level). A reject threshold that routes <= 20% of boundary cases to a human.
- Can extract discriminative anatomical biomarkers in the latent space despite variation in head shape and age between patients.
  *Test:* Grad-CAM maps focus on brain anatomy rather than the background or image border.
- Benchmark against the original "A ConvNet for the 2020s" paper [1], and against a simple baseline CNN on the same splits.

### Model Choice

Model: ConvNeXt [1]. Difficulty Band: Hard.

The model architecture and pipeline will be as follows. Where possible I use the same macro/micro design and hyperparameters as the paper [1], and I note where they have to change for this dataset and hardware.

1. Data preparation
- Data collection: supplied ADNI AD_NC dataset (`/home/groups/comp3710/ADNI`). The images are 2D greyscale JPEG slices of 256x240 px, with 20 slices per patient. Filenames follow `<patientID>_<sliceID>.jpeg`.
- Data split: Training/validation/testing. 
- Data balancing: the classes are close to balanced (about 48% AD in every split). 
- Data shuffling: training data is shuffled each epoch. Validation and test sets are not shuffled.
- Data labelling: labels come from the folder names (`AD` = 0, `NC` = 1).

2. Data Preprocessing
- Normalisation: Calculation the mean and standaard divation of the data set and normalise the values
- Regularisation (paper settings): weight decay 0.05, Stochastic Depth (drop path 0.1 for ConvNeXt-T), Label Smoothing 0.1, Layer Scale 1e-6.
- Batching: the paper uses a batch size of 4096, which does not fit on one GPU. I will use 32-128 with gradient accumulation and scale the learning rate linearly with batch size.
- Augmentation (paper settings): RandAugment (9, 0.5), Mixup 0.8, CutMix 1.0, Random Erasing 0.25. Mixup/CutMix blend two brains together and may hide anatomy, so I will test the model with and without them.

3. Network Architecture: ConvNeXt
- Macro design: stage compute ratio (3, 3, 9, 3) with channels (96, 192, 384, 768), about 28M parameters. A "patchify" stem (4x4 conv, stride 4) and separate 2x2 stride-2 downsampling layers between stages.
- Micro design: each block is a 7x7 depthwise conv -> LayerNorm -> 1x1 conv (expand x4, the inverted bottleneck) -> GELU -> 1x1 conv (project back), with Layer Scale and a residual connection.
- Activation Function: GELU, with only one activation per block.
- Normalisation: one LayerNorm per block instead of BatchNorm.
- Feature Extraction Method: hierarchical convolutional feature maps over 4 stages. Early stages detect edges and tissue boundaries, while later stages detect structures such as the ventricles and cortical atrophy. Global average pooling -> LayerNorm -> linear head with 2 outputs (AD, CN).

4. Training Procedure
- Optimiser: AdamW (uses momentum and adaptive learning rates to help find the minimum, with decoupled weight decay), weight decay 0.05.
- Learning Rate: paper base LR of 4e-3.
- Epochs: the paper uses 300. I will tune this on validation (expected 50-100).
- Cost/Loss Function: cross-entropy with label smoothing over 2 classes (binary classification: AD vs CN). The softmax output is used as the confidence score.

5. Evaluation Procedure:
- Training and validation: loss and accuracy plots, including "Training vs Validation Loss" (loss against epochs) and "Training vs Validation Accuracy" (accuracy against epochs). These plots help detect overfitting or underfitting.
- Performance Metrics: accuracy, per-class precision, recall, macro-F1, and the confusion matrix. For the open question: a reliability diagram, and accuracy against rejection-rate curves to choose a confidence threshold.
- Visualisation of Predictions: use activation maps (Grad-CAM) to determine how the model chose between the image classes.

6. Testing Procedure:
- Generalisability:
  - Evaluate against the test set (once, after model and threshold selection on validation).
  - Compare with a simple baseline CNN on the same test split.
- Compactness:
  - Determine the number of parameters, the memory footprint (peak GPU VRAM) and the storage size of the checkpoint.
- Inference Time:
  - Measure how long the model takes to process one input (or a batch of inputs) and output a prediction after training, and compare these results with the original benchmark paper (ConvNeXt-T: 774.7 img/s on a V100 at 224² [1]).

Dataset: ADNI dataset

The ADNI dataset provides structural MRI scans that can be preprocessed to include only the Alzheimer's disease and Cognitively Normal classes, accessible via the ADNI portal [3]. The full ADNI cohort has 591 subjects with Alzheimer's disease and 1479 cognitively normal subjects, with approximately a 50/50 split in sex and a varied age split.

The course subset used in this project contains 1,526 patients (743 AD, 783 CN). The data will be split into 3 sets, with each patient's data confined to a single split to avoid data leakage:

| Split | AD slices | NC slices | AD patients | NC patients |
|-------|----------:|----------:|------------:|------------:|
| Train | 8,840 | 9,460 | 442 | 473 |
| Validate (15% of the original train set, by patient) | 1,560 | 1,660 | 78 | 83 |
| Test | 4,460 | 4,540 | 223 | 227 |

### Course Concepts
Key course concepts that will help in building this model include:
- Convolutional Neural Networks (a class of deep learning models suited to image processing):
    - Inductive Biases: assumptions built into an algorithm, such as translational equivariance and local connectivity (connecting the pixels in a small region to a neuron), which come from convolution and help with image recognition. *Why it matters here:* patterns (an enlarged ventricle, a shrunken hippocampus) are local shapes that can appear at slightly different positions between patients. With only ~1,500 patients, a CNN needs less data than a Vision Transformer.
    - Kernel: a learned filter that starts from initial values and is then updated according to gradients to reduce classification loss/error.
    - Feature Map: the result of convolving a kernel with the input. It is a filtered version of the input that highlights the feature detected by the filter.
    - Padding: adds extra pixels to stop convolution from shrinking the feature map and to allow features to be detected at the edges of the input.
    - Stride: downsamples the input and captures more global features by skipping positions (the kernel moves more than one pixel at a time). ConvNeXt's patchify stem is a 4×4 conv with stride 4.
    - Layers: apply all the kernels to the input to get a set of feature maps, then apply new kernels to those feature maps, and so on. Each new set of feature maps is a new layer. Early layers capture fine-grained details like edges, middle layers detect textures and shapes, and deeper layers extract abstract features.
    - Pooling Layers: downsample the feature map by reducing spatial dimensions, using either average pooling (the average value in a region) or max pooling (the maximum value in a region). This reduces noise and improves memory efficiency. ConvNeXt uses global average pooling before the classifier head.
- Training:
    - Dropout: randomly disable neurons during training, forcing neurons to learn independently. This reduces overfitting and improves generalisability. ConvNeXt uses Stochastic Depth, a related method that randomly drops whole residual blocks.
    - Layer Normalisation: normalises across all channels at each spatial position in a layer, shifting the data back to a normalised distribution to improve convergence. ConvNeXt uses LayerNorm instead of BatchNorm.
- Classifiers: design a decision boundary between features that separates the classes (here, a linear head on pooled ConvNeXt features separates AD from CN).
- Neurons in Machine Learning:
    - Multi-Layer Perceptron (MLP): a network of fully connected neurons in hidden layers that takes an input and classifies it as an output. The 1×1 convs in a ConvNeXt block act as a per-pixel MLP.
    - Neuron: a node in a layer with multiple inputs, which it multiplies by a set of learned weights and sums. The sum is passed through an activation function (e.g. ReLU, GELU) that determines the neuron's output.
- Optimisation (a method for updating the weights in the network so the neurons become more accurate in their classifications):
    - Gradient Descent: calculate the derivative (gradient) of the cost function with respect to the weights, and step the weights in the opposite direction to move towards a minimum of the loss.
    - Backpropagation: send the error back through the layers using the chain rule, and update each neuron's weights to give better classifications.

How course concepts and literature support the choices:
- The paper [1] shows that a pure ConvNet with these modernised design choices matches or beats the Swin Transformer on ImageNet at similar compute (ConvNeXt-T 82.1% vs Swin-T 81.3%). It keeps the convolution inductive biases above, which matter more on a small medical dataset than on ImageNet.
- Prior ConvNeXt work on Alzheimer's MRI suggests the 80% target is achievable:
  - ConvNeXt-Base reaches 98.37% accuracy (AUC 0.992) on coronal slices, against about 92% on axial and sagittal slices [5], [8]. However, this was on a different, small and imbalanced dataset (AlzaSet: 79 subjects, 63 AD and 16 NC) [9].

### Preliminary Feasibility Evidence
Smoke test: I tested the baseline code from the paper's official repository [2], starting from ImageNet-1K pre-trained initial weights and fine-tuning on my ADNI dataset on a CPU. This checks that the data pipeline and training loop work end to end. The only code changes were adding an `ADNI` dataset option.

```bash
nohup python -u main.py \
  --model convnext_tiny --data_set ADNI \
  --data_path "data/ADNI/AD_NC/train" \
  --eval_data_path "data/ADNI/AD_NC/validation" \
  --finetune https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth \
  --epochs 11 --warmup_epochs 1 \
  --lr 1e-4 --min_lr 1e-6 \
  --batch_size 16 --update_freq 4 \
  --input_size 128 \
  --drop_path 0.1 --mixup 0 --cutmix 0 \
  --num_workers 4 --device cpu \
  --output_dir ./out_adni_cpu --log_dir ./out_adni_cpu \
  --auto_resume true \
  > train_cpu.log 2>&1 &
```

Monitoring it:
- `tail -f train_cpu.log` shows progress.

Test:
```bash
python main.py --model convnext_tiny --data_set ADNI --input_size 128 --device cpu \
  --eval true --resume out_adni_cpu/checkpoint-best.pth \
  --eval_data_path "data/ADNI/AD_NC/test"
```

Results:

| Measurement | Value |
|---|---|
| Parameters | 27.8 M |
| Training time (11 epochs, CPU) | 4 h 52 min |
| Best validation accuracy | 92.8% (epoch 9) |
| Test accuracy (9,000 slices) | **75.6%** (loss 0.628) |
| Test AD recall / CN recall (estimated*) | 57% / 94% |
| CPU inference | 3.7 s per batch of 96 at 128² ( 39 ms per slice) |

<details>
<summary>Raw test log</summary>

```
Test:  [ 0/94]  eta: 0:04:23  loss: 2.4426 (2.4426)  acc1: 4.1667 (4.1667)  acc5: 100.0000 (100.0000)  time: 2.8071  data: 0.5117
Test:  [10/94]  eta: 0:03:50  loss: 1.9510 (1.8226)  acc1: 26.0417 (25.9470)  acc5: 100.0000 (100.0000)  time: 2.7448  data: 0.0480
Test:  [20/94]  eta: 0:03:20  loss: 1.8853 (1.9073)  acc1: 20.8333 (23.8095)  acc5: 100.0000 (100.0000)  time: 2.7022  data: 0.0013
Test:  [30/94]  eta: 0:02:52  loss: 1.4274 (1.4033)  acc1: 33.3333 (44.3884)  acc5: 100.0000 (100.0000)  time: 2.6726  data: 0.0023
Test:  [40/94]  eta: 0:02:25  loss: 0.1186 (1.0981)  acc1: 96.8750 (56.6057)  acc5: 100.0000 (100.0000)  time: 2.6954  data: 0.0036
Test:  [50/94]  eta: 0:01:58  loss: 0.0801 (0.9719)  acc1: 97.9167 (61.8056)  acc5: 100.0000 (100.0000)  time: 2.6797  data: 0.0031
Test:  [60/94]  eta: 0:01:31  loss: 0.1118 (0.8549)  acc1: 91.6667 (66.4447)  acc5: 100.0000 (100.0000)  time: 2.6258  data: 0.0023
Test:  [70/94]  eta: 0:01:04  loss: 0.1881 (0.7621)  acc1: 91.6667 (70.2758)  acc5: 100.0000 (100.0000)  time: 2.6950  data: 0.0022
Test:  [80/94]  eta: 0:00:40  loss: 0.1339 (0.6869)  acc1: 95.8333 (73.3282)  acc5: 100.0000 (100.0000)  time: 3.4167  data: 0.0033
Test:  [90/94]  eta: 0:00:14  loss: 0.1407 (0.6414)  acc1: 95.8333 (75.0801)  acc5: 100.0000 (100.0000)  time: 6.9576  data: 0.0063
Test:  [93/94]  eta: 0:00:03  loss: 0.1424 (0.6276)  acc1: 94.7917 (75.5667)  acc5: 100.0000 (100.0000)  time: 7.4037  data: 0.0063
Test: Total time: 0:05:47 (3.7014 s / it)
Acc@1 75.567 Acc@5 100.000 loss 0.628
Accuracy of the network on 9000 test images: 75.56667%
```
</details>

![Training curves](images/training_curves.png)

*Smoke-test training curves. The orange "Test" line is the validation split.*

![Grad-CAM stage 3](images/gradcam_examples_stage3.png)

*Grad-CAM (stage 3) on test slices: evidence for AD and for NC.*

What this shows:
1. The pipeline works end to end and the model is learning: validation loss falls from 0.42 to 0.20 over 11 epochs.
2. Test accuracy (75.6%) is **below the 80% target**. Most errors are AD scans predicted as CN (false negatives), which is the opposite to the desired outcome.
4. Grad-CAM often highlights the ventricles and medial temporal region (plausible AD biomarkers). Some maps also highlight the image border or background, which suggests shortcut learning.

### Risks
The key risks are:
- The model may take a long time to train, leaving too little time to fine-tune the parameters to meet the minimum accuracy requirement. The paper trains for 300 epochs, and the CPU smoke test took ~5 h for 11 epochs at 128². *Mitigation:* train on GPU, start from ImageNet weights as tests, and run short test before long runs.
- The smoke test is below 80% on test and is biased towards false negatives. *Mitigation:* train at 224² for longer, aggregate the 20 slices of each patient into one patient-level prediction and set the decision threshold to favour AD recall.
- The validation set may not represent the test set. *Mitigation:* investigate the split.
- I have not yet tested running a Swin Transformer, so I may not be able to benchmark against the Swin Transformer myself. *Mitigation:* use the paper's published Swin-T numbers for comparison. The task sheet only requires a simple baseline CNN, which is lower risk.

### Computing Budget
There are three compute options:
- RTX 3080 (24/7 access)
- RTX2040 (lab computers, limited access)
- Rangpur A100 (dependent on demand and queue times)

The "A ConvNet for the 2020s" foundation paper measured inference throughput on V100 GPUs and training memory with a per-GPU batch size of 32. ConvNeXt-T runs at 1943.5 img/s vs 1325.6 img/s for Swin-T (+47%), and up to +49% for larger models.

Therefore, the RTX 3080 will be used for development and testing, as it is easy to access. Initial benchmark tests (ConvNeXt vs baseline: VRAM, training time, inference latency) will also be run on the RTX 3080 so the numbers are comparable.

### Next Planned Experiment
The next planned experiment is to modify the original paper's ConvNeXt to process ADNI MRI scans, combining techniques from:
- Basereh et al., ConvNeXt-driven detection of AD across anatomical planes (*Diagnostics*, 2025) [5]
- Vyas, NeuroScan / adni_code (GitHub) [6]
- Jin et al., CA-ConvNeXt (Intelligence Science IV, 2022) [7]
- Basereh et al., bioRxiv preprint of [5] [8]

I will write my own ConvNeXt in `modules.py` (following the paper's architecture) along with `dataset.py`, `train.py` and `predict.py`. I will then run this model without pre-training on the RTX 3080, and compare it with the ImageNet-pretrained version from the smoke test. Hyperparameters and parameters will be fine-tuned from there to reach the required minimum test accuracy of 0.8.

If time allows I would like to test how the data affects the model by using the AlzaSet dataset [9] (the dataset behind [5]) to check whether including coronal, sagittal and axial MRI slices improves model accuracy. This would indicate which angle of brain slice is most informative for Alzheimer's detection.

### Fallback
If the ConvNeXt project takes too long to train, or fails to produce viable results, I will revert to project 2.2 "Segmentation: Anatomical Contouring and Radiotherapy Planning" with U-Net (2D U-Net / Improved U-Net on OASIS is Easy difficulty, capped at 10/20 for implementation).

The U-Net architecture is also based on convolution, the same as ConvNeXt. It contains an encoder, a bottleneck and a decoder, forming a U-shaped structure. 

This project similarly looks at brain tissue slices. However, it focuses on delineating cortical grey matter, white matter and cerebrospinal fluid across 2D slices, whereas ConvNeXt aims to classify the entire image as either Cognitively Normal or Alzheimer's.

The similar convolution concepts and datasets make this a viable fallback, especially as a similar project was completed in Demo 2.

-----------------------------------------

## Dependencies 

## Reproducibility

## Examples

### Inputs

### Outputs

### Visualisations

# Open Research Dilemma

## Quantitative Benchmarking

## Computational Efficiency

## Qualitative Error

## Engineering Recommendations

## Artificial Intelligence Usage Disclosure

## References

[1] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 11976–11986. [Online]. Available: https://arxiv.org/abs/2201.03545

[2] Meta AI Research, "ConvNeXt: official PyTorch implementation," GitHub repository. [Online]. Available: https://github.com/facebookresearch/ConvNeXt

[3] Alzheimer's Disease Neuroimaging Initiative (ADNI). [Online]. Available: http://adni.loni.usc.edu

[4] G. H. Huynh, "Alzheimers-MRI-ConvNext: PyTorch implementation of ConvNeXt for Alzheimer's MRI classification," GitHub repository. [Online]. Available: https://github.com/hunghuynh1110/Alzheimers-MRI-ConvNext

[5] M. Basereh, M. A. Abikenari, S. Sadeghzadeh, T. Dunn, R. Freichel, P. Siddarth, D. Ghahremani, H. Lavretsky, and V. P. Buch, "ConvNeXt-driven detection of Alzheimer's disease: A benchmark study on expert-annotated AlzaSet MRI dataset across anatomical planes," *Diagnostics*, vol. 15, no. 23, Art. no. 2997, 2025, doi: 10.3390/diagnostics15232997. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC12691013/

[6] M. R. Vyas, "NeuroScan (adni_code)," GitHub repository. [Online]. Available: https://github.com/MeetRVyas/adni_code

[7] W. Jin, Y. Yin, J. Bai, and H. Zhen, "CA-ConvNeXt: Coordinate attention on ConvNeXt for early Alzheimer's disease classification," in *Intelligence Science IV*, IFIP Advances in Information and Communication Technology. Cham, Switzerland: Springer, 2022, pp. 450–457, doi: 10.1007/978-3-031-14903-0_48. [Online]. Available: https://link.springer.com/chapter/10.1007/978-3-031-14903-0_48

[8] M. Basereh *et al.*, "ConvNeXt-driven detection of Alzheimer's disease: A benchmark study on expert-annotated AlzaSet MRI dataset across anatomical planes," bioRxiv preprint, Jul. 2025, doi: 10.1101/2025.07.10.664260. [Online]. Available: https://www.biorxiv.org/content/10.1101/2025.07.10.664260v1.full

[9] M. Basereh, "AlzaSet: MRI dataset for 'ConvNeXt-driven Alzheimer's detection: A benchmark study on expert-annotated AlzaSet MRI dataset across anatomical planes'," Zenodo, version v2, 2025, doi: 10.5281/zenodo.17084199. [Online]. Available: https://zenodo.org/records/17084199
