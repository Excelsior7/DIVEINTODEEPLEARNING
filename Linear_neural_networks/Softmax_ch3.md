```python
import torch
import torchvision
```

***
### SOFTMAX CLASSIFICATION MODEL IMPLEMENTATION 
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

Using the FashionMNIST dataset, each image has tensor shape (1,28,28).    
Reshaping the tensor shape to (784), thus regarding the image as a feature vector
help us define the parameter W with shape    
(target's dimension, feature vector's dimension) = (10,784).


```python
W = torch.distributions.normal.Normal(0,1).sample(sample_shape=(10,784));
b = torch.distributions.normal.Normal(0,1).sample(sample_shape=(10,));
W.requires_grad = True;
b.requires_grad = True;
```

***
### *MODEL*

logits(X) = $ \left[\begin{array} {ccc}  \vdots&\vdots&\vdots&\vdots \\ o_{i,0} & o_{i,1} & \cdots & o_{i,9} \\ \vdots&\vdots&\vdots&\vdots \end{array} \right]$


```python
def logits(X):
    return torch.matmul(X.reshape(-1,784),torch.transpose(W,0,1)) + b;
```

***
### *Log-sum-exp trick*

- The softmax function can leads to overflow problems due to the exponential terms.    
In order to avoid overflow problems we will use the [log-sum-exp trick](https://en.wikipedia.org/wiki/LogSumExp).
But this trick can itself leads to underflow problems, which can be solved by taking the log.

- Given an example $(x^{(i)}, y^{(i)})\  s.t.\  x^{(i)} \in \mathbb{R}^{784}, y^{(i)} \in \{0, \cdots, 9\}$ what interest us is to compute the following quantity:

${\displaystyle softmax(\mathbf{o}_{i})_{y^{(i)}} = \frac{exp(o_{i,y^{(i)}})}{\sum_{j=0}^{9} exp(o_{i,j})} }$

- So let's define $x_{i}^{*} = max_k \   o_{i,k} $


- Then, ${\displaystyle softmax(\mathbf{o}_{i})_{y^{(i)}} = \frac{exp(o_{i,y^{(i)}} - x_{i}^{*})}{\sum_{j=0}^{9} exp(o_{i,j} - x_{i}^{*})} }$, this remove the overflow problem without changing the probability!

- Now, in order to avoid underflow problem, I take the log of this expression,

${\displaystyle log(softmax(\mathbf{o}_{i})_{y^{(i)}}) = log(\frac{exp(o_{i,y^{(i)}} - x_{i}^{*})}{\sum_{j=0}^{9} exp(o_{i,j} - x_{i}^{*})}) = o_{i,y^{(i)}} - x_{i}^{*} - log(\sum_{j=0}^{9} exp(o_{i,j} - x_{i}^{*}))}$

- $ - log(softmax(\mathbf{o}_{i})_{y^{(i)}}) $ is just the **cross entropy loss function** over one example!


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
sgd_optimizer = torch.optim.SGD((W,b), lr=0.01);
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
### TRAINING


```python
def softmaxTraining(training_dataset, model, loss, optimizer, num_epochs):
    
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
softmaxTraining(training_dataset, logits, crossEntropy, sgd_optimizer, 5);
```

    Prediction accuracy  tensor(0.6988)
    Prediction accuracy  tensor(0.7716)
    Prediction accuracy  tensor(0.7863)
    Prediction accuracy  tensor(0.7944)
    Prediction accuracy  tensor(0.7980)


***
### TEST


```python
def softmaxTest(test_dataset, model):
    
    test_metrics = Accumulator(2);
    
    for X,y in test_dataset:
        test_metrics.add(prediction(model(X),y), y.numel());
    
    print("Prediction accuracy on test", test_metrics.getData()[0] / test_metrics.getData()[1]);
```


```python
softmaxTest(test_dataset, logits);
```

    Prediction accuracy on test tensor(0.7943)

