```python
import torch
import torch.nn as nn
import numpy as np
```

***
### BIDIRECTIONAL RECURRENT NEURAL NETWORK IMPLEMENTATION FROM SCRATCH
***

![jpg](../plots/BRNN_fig1.JPG);

### *DATA*


```python
num_data = 5000;
n = 500;

X = torch.floor(10*torch.rand((num_data, 6)));
Y = torch.empty((num_data,));

for i in range(num_data):
    seqf, seqb = "", "";
    for u in range(3):
        seqf = seqf + str(int(X[i,0:3][u].item()));
        seqb = seqb + str(int(X[i,3:6][2-u].item()));
        
    out_3 = 1 if int(seqf) < 500 else 0;
    out_5 = 1 if int(seqb) > 500 else 0;
    
    Y[i] = out_3*out_5;
```


```python
num_training_data = int(np.floor(num_data*0.75));

X_train = X[0:num_training_data];
Y_train = Y[0:num_training_data];

X_test = X[num_training_data:];
Y_test = Y[num_training_data:];
```

***
### *DATA LOADER*


```python
def XYdataLoader(X, Y, batch_size, shuffle):

    dataset = torch.utils.data.TensorDataset(X,Y);
    
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=2);
```


```python
batch_size = 5000;

dataset_train = XYdataLoader(X_train, Y_train, batch_size, True);
dataset_test = XYdataLoader(X_test, Y_test, batch_size, False);
```

***
### *MODEL*


```python
class BRNN(nn.Module):
    def __init__(self, x_size, h_size, q_size):
        super().__init__();
            
        self.fW_xh = nn.Parameter(torch.randn(x_size, h_size));
        self.fW_hh = nn.Parameter(torch.randn(h_size, h_size));
        self.fb_h = nn.Parameter(torch.zeros(h_size));
        
        self.bW_xh = nn.Parameter(torch.randn(x_size, h_size));
        self.bW_hh = nn.Parameter(torch.randn(h_size, h_size));
        self.bb_h = nn.Parameter(torch.zeros(h_size));
        
        self.fW_hq = nn.Parameter(torch.randn(h_size, q_size));
        self.fb_q = nn.Parameter(torch.zeros(q_size));
        
        self.bW_hq = nn.Parameter(torch.randn(h_size, q_size));
        self.bb_q = nn.Parameter(torch.zeros(q_size));
        

    def forward(self, fX, bX):
        
        fH, bH = None, None;
        RL = torch.nn.ReLU();
        
        ## A always refers to the inputs (X,H).
        ## B always refers to the parameters (W).
        matmul = lambda A,B: torch.matmul(A,B) if A is not None else 0;
        
        for i in range(len(fX)):
            fH = RL(matmul(fX[i], self.fW_xh) + matmul(fH, self.fW_hh) + self.fb_h);
        
        for i in range(len(bX)):
            bH = RL(matmul(bX[i], self.bW_xh) + matmul(bH, self.bW_hh) + self.bb_h);
            
        out_3 = matmul(fH, self.fW_hq) + self.fb_q;
        out_5 = matmul(bH, self.bW_hq) + self.bb_q;
        
        return out_3 * out_5;
```

***
### *LOSS*


```python
def loss(Y_hat, Y):    
    return ((Y_hat-Y)**2).mean();
```

***
### *TRANSFORMATIONS*


```python
def XTransform(X, flip=False):

    if flip == True:
        X = torch.flip(X,dims=(1,));
        
    X = torch.transpose(X,0,1);
    X = X.reshape(X.shape[0], -1, 1);
    
    return X;
```

***
### *GRADIENT CLIPPING*

I use equation **(9.5.3)** defined in https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html#gradient-clipping and i define:    
**theta = alpha*torch.numel(param.grad)** where **alpha** is a hyperparameter.


```python
def gradientClipping(brnn, alpha=1):
    for param in brnn.parameters():
        if param.requires_grad:
            norm = torch.norm(param.grad.flatten());
            theta = alpha*param.numel();
            
            if norm > theta:
                param.grad *= theta/norm;
```

***
### *TRAINING*


```python
def trainBRNN(brnn, dataset, loss, optimizer, num_epochs, alpha=1):
    
    brnn.train();
    
    for epoch in range(num_epochs):
        fX_test, bX_test, Y_test = None, None, None;

        for X,Y in dataset:
            fX, bX = XTransform(X[:,0:3]), XTransform(X[:,3:6], flip=True);
            l = loss(brnn(fX, bX), Y);

            with torch.no_grad():
                l.backward();
                gradientClipping(brnn, alpha);
                optimizer.step();
                optimizer.zero_grad();

            if fX_test == None and bX_test == None and Y_test == None:
                fX_test, bX_test, Y_test = fX, bX, Y;
            
        print(f'Training loss {loss(brnn(fX_test, bX_test), Y_test)}');
        print(f'Epoch {epoch}');                        
        
    return brnn;
```


```python
brnn = BRNN(1, 32, 1);
```


```python
for i in range(2):
    optimizer = torch.optim.SGD(brnn.parameters(), lr=0.1/10**i);    
    brnn_trained = trainBRNN(brnn, dataset_train, loss, optimizer, (i+1)*100);
```

    Training loss 6.742625349323981e+16
    Epoch 0
    Training loss 13166215168.0
    Epoch 1
    Training loss 57544688.0
    Epoch 2
    Training loss 2221302016.0
    Epoch 3
    ...
    Training loss 0.18849396705627441
    Epoch 196
    Training loss 0.18849390745162964
    Epoch 197
    Training loss 0.1884937584400177
    Epoch 198
    Training loss 0.18849366903305054
    Epoch 199


***
### *TESTING*


```python
def testBRNN(brnn, dataset):
    
    L = [];
    
    for X,Y in dataset:
        fX, bX = XTransform(X[:,0:3]), XTransform(X[:,3:6], flip=True);
        l = loss(brnn(fX, bX), Y);
        
        L.append(l);
        
    print(f'mean loss at testing time {torch.tensor(L).mean()}');
```


```python
testBRNN(brnn_trained, dataset_test);
```

    mean loss at testing time 0.20345640182495117

