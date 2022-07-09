```python
import torch 
import torch.nn as nn
import numpy as np
```

***
### VGG BLOCKS NETWORK CLASSIFICATION MODEL IMPLEMENTATION 
***

### *BLOCK GENERATING FUNCTION*

PARAMETERS : 
* num_conv $ \in \mathbb{N}_1 $
* in_channels $ \in \mathbb{N}_1 $
* out_channels $ \in \mathbb{N}_1 $
* conv_kernel_window_size $ \{(k_h, k_w) | k_h \in \mathbb{N}_1, k_w \in \mathbb{N}_1\} $ 
    + s.t. $k_h$, $k_w$ denote the height and the width of the kernel window respectively.
    
NOTE :
Using the section (6.3.1) in https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html we can define the padding size given the kernel window size so as to keep the same shape between the input and the output of a convolutional layer. (This is implemented in the "paddingHomogeneity" function)


```python
def vggBlock(num_conv, in_channels, out_channels, conv_kernel_window_size):
    
    padding = paddingHomogeneity(conv_kernel_window_size[0], conv_kernel_window_size[1]);

    layers_in_block = [];
    
    for _ in range(num_conv):
        layers_in_block.append(nn.Conv2d(in_channels, out_channels, conv_kernel_window_size, padding = padding));
        layers_in_block.append(nn.ReLU());
        
        in_channels = out_channels;
    
    layers_in_block.append(nn.MaxPool2d(kernel_size = 2, stride = 2));
    
    return nn.Sequential(*layers_in_block);
```


```python
def paddingHomogeneity(k_h,k_w):
    
    if k_h % 2 == 1 and k_w % 2 == 1:
        padding = ((k_h - 1)/2,(k_w - 1)/2);
        
    elif k_h % 2 == 1 and k_w % 2 == 0:
        padding = ((k_h - 1)/2,(k_h - 1)/2, np.floor((k_w - 1)/2), np.ceil((k_w - 1)/2));
        
    elif k_h % 2 == 0 and k_w % 2 == 1:
        padding = (np.floor((k_h - 1)/2), np.ceil((k_h - 1)/2), (k_w - 1)/2,(k_w - 1)/2);
        
    else:
        padding = (np.floor((k_h - 1)/2), np.ceil((k_h - 1)/2), np.floor((k_w - 1)/2), np.ceil((k_w - 1)/2));
        
    return [int(padding[i]) for i in range(len(padding))];
```


```python
vggBlock(4,1,16,(5,3))
```




    Sequential(
      (0): Conv2d(1, 16, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1))
      (1): ReLU()
      (2): Conv2d(16, 16, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1))
      (3): ReLU()
      (4): Conv2d(16, 16, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1))
      (5): ReLU()
      (6): Conv2d(16, 16, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1))
      (7): ReLU()
      (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )



***
### *VGG BLOCKS NETWORK*


```python
# (num_conv, out_channels, conv_kernel_window_size);
vgg_blocks_arch = [(1,64,(3,3)),(1,128,(3,3)),(2,256,(3,3)),(2,512,(3,3)),(2,512,(3,3))]; 
```


```python
def vggNet(vgg_arch, in_channels, input_shape):
    
    vgg_blocks = [];
    in_height = input_shape[0];
    in_width = input_shape[1];
    
    for num_conv, out_channels, conv_kernel_window_size in vgg_blocks_arch:
        vgg_blocks.append(vggBlock(num_conv, in_channels, out_channels, conv_kernel_window_size));
        
        in_height = int(in_height/2) if in_height % 2 == 0 else int((in_height-1)/2);
        in_width = int(in_width/2) if in_width % 2 == 0 else int((in_width-1)/2);
        
        in_channels = out_channels;
    
    return nn.Sequential(*vgg_blocks, nn.Flatten(),
                         nn.Linear(out_channels*in_height*in_width, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 10)
                        );
```


```python
net = vggNet(vgg_blocks_arch, 1, (224,224));
```


```python
X = torch.randn(size=(1,1,224,224));
for blk in net:
    X = blk(X);
    print(blk.__class__.__name__,'output shape:\t', X.shape);
```

    Sequential output shape:	 torch.Size([1, 64, 112, 112])
    Sequential output shape:	 torch.Size([1, 128, 56, 56])
    Sequential output shape:	 torch.Size([1, 256, 28, 28])
    Sequential output shape:	 torch.Size([1, 512, 14, 14])
    Sequential output shape:	 torch.Size([1, 512, 7, 7])
    Flatten output shape:	 torch.Size([1, 25088])
    Linear output shape:	 torch.Size([1, 4096])
    ReLU output shape:	 torch.Size([1, 4096])
    Dropout output shape:	 torch.Size([1, 4096])
    Linear output shape:	 torch.Size([1, 4096])
    ReLU output shape:	 torch.Size([1, 4096])
    Dropout output shape:	 torch.Size([1, 4096])
    Linear output shape:	 torch.Size([1, 10])

