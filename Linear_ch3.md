```python
import torch
```

***
### LINEAR REGRESSION MODEL IMPLEMENTATION 
***

**DESIGN MATRIX : X**      
**LABELS : Y**


- Data generating process : $ y = Ax + b + \epsilon$

    - input : $ x \in \mathbb{R}^{2}$
    - output : $ y \in \mathbb{R}$
    - $\epsilon  \sim \mathcal{N}(0,1)$ 
    
    - Parameters :
        - $ A \in \mathbb{R}^{1 \times 2}$
        - $ b \in \mathbb{R}$

***
### *DATA*


```python
A = torch.tensor([4.0,7.0]);
b = torch.tensor(7.0);
```


```python
X = torch.rand(100,2, dtype=torch.float32);
Y = (X @ A) + b + torch.distributions.normal.Normal(0,1).sample((100,));
Y = Y.reshape((X.shape[0],1));

X[:10],Y[:10]
```




    (tensor([[0.3853, 0.8791],
             [0.3279, 0.0623],
             [0.2667, 0.7593],
             [0.0655, 0.5975],
             [0.8937, 0.6758],
             [0.0305, 0.5711],
             [0.3225, 0.1309],
             [0.9858, 0.9686],
             [0.6177, 0.1667],
             [0.7765, 0.4532]]),
     tensor([[15.5063],
             [10.5544],
             [13.5841],
             [ 9.9937],
             [15.0731],
             [11.0389],
             [ 8.1411],
             [19.1830],
             [11.4804],
             [12.9250]]))




```python
batch_size = 10;

tensor_dataset = torch.utils.data.TensorDataset(X,Y);
data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size, shuffle=True);
```

***
### *MODEL*


```python
model = torch.nn.Linear(2,1);
```

***
### *LOSS*


```python
loss = torch.nn.MSELoss(); 
```

***
### *OPTIMIZATION ALGORITHM*


```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01);
```

***   
### TRAINING


```python
num_epochs = 10;

for epoch in range(num_epochs):
    for X_batch,Y_batch in data_loader:
        model_loss = loss(model(X_batch), Y_batch);
        model_loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    
    print(f'epoch #{epoch+1}, loss {loss(model(X), Y)}');
```

    epoch #1, loss 84.30823516845703
    epoch #2, loss 45.919837951660156
    epoch #3, loss 25.318326950073242
    epoch #4, loss 14.197885513305664
    epoch #5, loss 8.259031295776367
    epoch #6, loss 5.067922592163086
    epoch #7, loss 3.337728261947632
    epoch #8, loss 2.405332088470459
    epoch #9, loss 1.898695945739746
    epoch #10, loss 1.618688941001892

