```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary as summary
```

***
### GOOGLENET CLASSIFICATION MODEL IMPLEMENTATION 
***

PARAMETERS : 
* in_channels $ \in \mathbb{N}_1 $
* out_channels = $ \{(out_1, out_2, out_3, out_4) | out_i \in \mathbb{N}_1, i=\{1,2,3,4\} \} $

***
### *GoogLeNet Block*


```python
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__();        
        
        out_1, out_2, out_3, out_4 = out_channels;
        
        self.p1 = nn.Conv2d(in_channels, out_1, kernel_size=1);
        
        self.p2_1 = nn.Conv2d(in_channels, out_2, kernel_size=1);
        self.p2_2 = nn.Conv2d(out_2, out_2, kernel_size=3, padding=1);
        
        self.p3_1 = nn.Conv2d(in_channels, out_3, kernel_size=1);
        self.p3_2 = nn.Conv2d(out_3, out_3, kernel_size=5, padding=2);
        
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1);
        self.p4_2 = nn.Conv2d(in_channels, out_4, kernel_size=1);
    
    def forward(self, X):
        c1 = F.relu(self.p1(X));
        c2 = F.relu(self.p2_2(F.relu(self.p2_1(X))));
        c3 = F.relu(self.p3_2(F.relu(self.p3_1(X))));
        c4 = F.relu(self.p4_2(self.p4_1(X)));
        
        return torch.cat((c1,c2,c3,c4), dim=1);
```

***
### GoogLeNet Model


```python
# X = (5,3,224,224);
# => in_channels = 3;
```


```python
block_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1));

# out_channels Block 1 = 64;
```


```python
block_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1));

# out_channels Block 2 = 64;
```


```python
block_3 = nn.Sequential(InceptionBlock(64, (16, 32, 32, 16)),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1));

# out_channels Block 3 = 16+32+32+16 = 96;
```


```python
block_4 = nn.Sequential(InceptionBlock(96, (32, 64, 64, 32)),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1));

# out_channels Block 4 = 32+64+64+32 = 192;
```


```python
block_5 = nn.Sequential(InceptionBlock(192, (64, 128, 128, 64)),
                        nn.AdaptiveMaxPool2d((1,1)));

# out_channels Block 5 = 64+128+128+64 = 384;
```


```python
net = nn.Sequential(block_1, block_2, block_3, block_4, block_5, 
                          nn.Flatten(), nn.Linear(384, 10));
```


```python
X = torch.randn(size=(5,3,224,224));
for blk in net:
    X = blk(X);
    print(blk.__class__.__name__,'output shape:\t', X.shape);
```

    Sequential output shape:	 torch.Size([5, 64, 56, 56])
    Sequential output shape:	 torch.Size([5, 64, 28, 28])
    Sequential output shape:	 torch.Size([5, 96, 14, 14])
    Sequential output shape:	 torch.Size([5, 192, 7, 7])
    Sequential output shape:	 torch.Size([5, 384, 1, 1])
    Flatten output shape:	 torch.Size([5, 384])
    Linear output shape:	 torch.Size([5, 10])

