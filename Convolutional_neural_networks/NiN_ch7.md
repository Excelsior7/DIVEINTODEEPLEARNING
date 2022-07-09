```python
import torch 
import torch.nn as nn
```

***
### NETWORK IN NETWORK CLASSIFICATION MODEL IMPLEMENTATION 
***

PARAMETERS : 
* in_channels $ \in \mathbb{N}_1 $
* out_channels $ \in \mathbb{N}_1 $
* conv_kernel_window_size $ \{(k_h, k_w) | k_h \in \mathbb{N}_1, k_w \in \mathbb{N}_1\} $ 
    + s.t. $k_h$, $k_w$ denote the height and the width of the kernel window respectively.

***
### *NiN Block*


```python
def NiNBlock(in_channels, out_channels, conv_kernel_window_size, stride, padding):
    l1 = nn.Conv2d(in_channels, out_channels, conv_kernel_window_size, stride, padding);
    l2 = nn.Conv2d(out_channels, out_channels, kernel_size=1);
    l3 = nn.Conv2d(out_channels, out_channels, kernel_size=1);
    
    return nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3, nn.ReLU());
```

***
### *NiN model*


```python
#(out_channels, conv_kernel_window_size, stride, padding);
NiN_model_arch = [(96,(11,11),1,5), (256,(5,5),1,2), (384,(3,3),1,1), (10,(3,3),1,1)];
```


```python
def NiN(in_channels, NiN_model_arch):
    
    NiN_blocks = [];
    num_of_NiN_blocks = len(NiN_model_arch);
    
    for i, NiN_block in enumerate(NiN_model_arch):
        out_channels, conv_kernel_window_size, stride, padding = NiN_block;
        NiN_blocks.append(NiNBlock(in_channels, out_channels, conv_kernel_window_size, stride, padding));

        if i+1 < len(NiN_model_arch):
            NiN_blocks.append(nn.MaxPool2d(kernel_size=3, stride=2));

        in_channels = out_channels;
                 
    return nn.Sequential(*NiN_blocks, nn.AdaptiveMaxPool2d((1,1)));
```


```python
net = NiN(3, NiN_model_arch);
```


```python
X = torch.randn(size=(5,3,224,224));
for blk in net:
    X = blk(X);
    print(blk.__class__.__name__,'output shape:\t', X.shape);
```

    Sequential output shape:	 torch.Size([5, 96, 224, 224])
    MaxPool2d output shape:	 torch.Size([5, 96, 111, 111])
    Sequential output shape:	 torch.Size([5, 256, 111, 111])
    MaxPool2d output shape:	 torch.Size([5, 256, 55, 55])
    Sequential output shape:	 torch.Size([5, 384, 55, 55])
    MaxPool2d output shape:	 torch.Size([5, 384, 27, 27])
    Sequential output shape:	 torch.Size([5, 10, 27, 27])
    AdaptiveMaxPool2d output shape:	 torch.Size([5, 10, 1, 1])


![png](plots/NiN_scheme.png) 
