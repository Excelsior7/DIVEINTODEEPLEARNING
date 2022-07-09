```python
import torch
import torchvision
import torch.nn as nn
```

***
### MULTILAYER PERCEPTRONS CLASSIFICATION MODEL IMPLEMENTATION 
***

### *DATA*
[FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)


```python
def datasets(dataset_name, root, transform, batch_size, num_workers, target_transform=None, shuffle=True):
    training_dataset = getattr(torchvision.datasets, dataset_name)(root=root,
                                                    train=True,
                                                    download=True,
                                                    transform=transform, 
                                                    target_transform=target_transform);
    test_dataset = getattr(torchvision.datasets, dataset_name)(root=root,
                                                    train=False,
                                                    download=True,
                                                    transform=transform,
                                                    target_transform=target_transform);
    
    return (torch.utils.data.DataLoader(dataset=training_dataset, 
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers),
           torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers));
```


```python
dataset_name = "FashionMNIST";
root = "/home/excelsior/Desktop/d2l-en/models_implementation/";
transform = torchvision.transforms.ToTensor();
batch_size = 256;
num_workers=2;

training_dataset, test_dataset = datasets(dataset_name, root, transform, batch_size, num_workers);
```

***
### *PARAMETERS*


```python
dim_inputs, dim_outputs, dim_hidden_units = 784, 10, 256;

W1 = torch.distributions.normal.Normal(0,1).sample(sample_shape=(dim_inputs,dim_hidden_units));
b1 = torch.distributions.normal.Normal(0,1).sample(sample_shape=((dim_hidden_units,)));
W2 = torch.distributions.normal.Normal(0,1).sample(sample_shape=(dim_hidden_units,dim_outputs));
b2 = torch.distributions.normal.Normal(0,1).sample(sample_shape=((dim_outputs,)));

W1.requires_grad = True;
b1.requires_grad = True;
W2.requires_grad = True;
b2.requires_grad = True;

params = (W1,b1,W2,b2);
```

***
### *MODEL*


```python
def mlpLogits(X):
    X = X.reshape(-1,dim_inputs);
    H1L = relu(X@W1 + b1);
    OL = H1L@W2 + b2;

    return OL;
```


```python
def relu(HL):
    HL_dim_hidden_units = len(HL[0]);
    return torch.max(HL, torch.zeros(HL_dim_hidden_units));
```

***
### *Log-sum-exp trick*


```python
def logSumExpTrick(logits,y):

    logits_max_subtracted = logits - torch.max(logits, axis=1, keepdim=True).values;
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_max_subtracted), axis=1));
    
    return logits_max_subtracted[range(len(y)), y] - log_sum_exp;
```

***
### *LOSS*


```python
def crossEntropy(logits,y):
    return - torch.sum(logSumExpTrick(logits,y));
```

***
### *OPTIMIZATION ALGORITHM*


```python
sgd_optimizer = torch.optim.SGD((params), lr=0.0001);
```

***
### *PREDICTION ACCURACY*


```python
def prediction(logits,y):
        predictions = torch.argmax(logits, axis=1);
        is_prediction_correct = (predictions == y);
        
        return is_prediction_correct.sum();
```


```python
class Accumulator():
    
    def __init__(self,n):
        self.data = [0] * n;
        
    def add(self, *args):
        self.data = [a+b for a,b in zip(self.data, args)];
        
    def reset(self, r):
        self.data = [0] * r;
        
    def getData(self):
        return self.data;
        
```

***
### *TRAINING*


```python
def mlpTraining(training_dataset, model, loss, optimizer, num_epochs):
    
    metrics = Accumulator(2);
    
    for epoch in range(num_epochs):
        for X,y in training_dataset:
            l = loss(model(X),y);
            l.backward();
            optimizer.step();
            optimizer.zero_grad();
            
            metrics.add(prediction(model(X),y),y.numel());
            
        print("Prediction accuracy ", metrics.getData()[0] / metrics.getData()[1]);
        metrics.reset(2);
```


```python
mlpTraining(training_dataset, mlpLogits, crossEntropy, sgd_optimizer, 50);
```

    Prediction accuracy  tensor(0.6618)
    Prediction accuracy  tensor(0.7506)
    Prediction accuracy  tensor(0.7716)
    Prediction accuracy  tensor(0.7840)
    Prediction accuracy  tensor(0.7942)
    Prediction accuracy  tensor(0.7978)
    Prediction accuracy  tensor(0.8013)
    Prediction accuracy  tensor(0.8060)
    Prediction accuracy  tensor(0.8087)
    Prediction accuracy  tensor(0.8085)
    Prediction accuracy  tensor(0.8115)
    Prediction accuracy  tensor(0.8112)
    Prediction accuracy  tensor(0.8135)
    Prediction accuracy  tensor(0.8151)
    Prediction accuracy  tensor(0.8172)
    Prediction accuracy  tensor(0.8172)
    Prediction accuracy  tensor(0.8192)
    Prediction accuracy  tensor(0.8199)
    Prediction accuracy  tensor(0.8207)
    Prediction accuracy  tensor(0.8230)
    Prediction accuracy  tensor(0.8231)
    Prediction accuracy  tensor(0.8249)
    Prediction accuracy  tensor(0.8263)
    Prediction accuracy  tensor(0.8275)
    Prediction accuracy  tensor(0.8275)
    Prediction accuracy  tensor(0.8304)
    Prediction accuracy  tensor(0.8310)
    Prediction accuracy  tensor(0.8314)
    Prediction accuracy  tensor(0.8332)
    Prediction accuracy  tensor(0.8336)
    Prediction accuracy  tensor(0.8354)
    Prediction accuracy  tensor(0.8361)
    Prediction accuracy  tensor(0.8374)
    Prediction accuracy  tensor(0.8384)
    Prediction accuracy  tensor(0.8398)
    Prediction accuracy  tensor(0.8404)
    Prediction accuracy  tensor(0.8421)
    Prediction accuracy  tensor(0.8416)
    Prediction accuracy  tensor(0.8431)
    Prediction accuracy  tensor(0.8437)
    Prediction accuracy  tensor(0.8442)
    Prediction accuracy  tensor(0.8452)
    Prediction accuracy  tensor(0.8452)
    Prediction accuracy  tensor(0.8456)
    Prediction accuracy  tensor(0.8467)
    Prediction accuracy  tensor(0.8471)
    Prediction accuracy  tensor(0.8488)
    Prediction accuracy  tensor(0.8495)
    Prediction accuracy  tensor(0.8490)
    Prediction accuracy  tensor(0.8497)


***
### *TEST*


```python
def mlpTest(test_dataset, model):
    
    test_metrics = Accumulator(2);
    
    for X,y in test_dataset:
        test_metrics.add(prediction(model(X),y), y.numel());
    
    print("Prediction accuracy on test", test_metrics.getData()[0] / test_metrics.getData()[1]);
```


```python
mlpTest(test_dataset, mlpLogits)
```

    Prediction accuracy on test tensor(0.8155)



***
### MULTILAYER PERCEPTRONS CLASSIFICATION MODEL IMPLEMENTATION PYTORCH
***


### *MODEL*


```python
mlp = torch.nn.Sequential(nn.Flatten(),
                    nn.Linear(dim_inputs, dim_hidden_units),
                    nn.ReLU(),
                    nn.Linear(dim_hidden_units, dim_outputs));
```

***
### *LOSS*


```python
loss_nn = torch.nn.CrossEntropyLoss();
```

***
### *OPTIMIZATION ALGORITHM*


```python
optimizer_nn = torch.optim.SGD(mlp.parameters(), lr=0.001);
```

***
### *TRAINING*


```python
def mlpTrainingNN(training_dataset, model, loss, optimizer, num_epochs):
    
    metrics_nn = Accumulator(2);
    
    for epoch in range(num_epochs):
        for X,y in training_dataset:
            l = loss(mlp(X),y);
            l.backward();
            optimizer.step();
            optimizer.zero_grad();
            
            metrics_nn.add(prediction(model(X),y),y.numel());
            
        print("Prediction accuracy ", metrics_nn.getData()[0] / metrics_nn.getData()[1]);
        metrics_nn.reset(2);
```


```python
mlpTrainingNN(training_dataset, mlp, loss_nn, optimizer_nn, 10);
```

    Prediction accuracy  tensor(0.2589)
    Prediction accuracy  tensor(0.4403)
    Prediction accuracy  tensor(0.5622)
    Prediction accuracy  tensor(0.6302)
    Prediction accuracy  tensor(0.6445)
    Prediction accuracy  tensor(0.6494)


***
### *TEST*


```python
def mlpTestNN(test_dataset, model):
    
    test_metrics_nn = Accumulator(2);
    
    for X,y in test_dataset:
        test_metrics_nn.add(prediction(model(X),y), y.numel());
    
    print("Prediction accuracy on test", test_metrics_nn.getData()[0] / test_metrics_nn.getData()[1]);
```


```python
mlpTestNN(test_dataset, mlp);
```
