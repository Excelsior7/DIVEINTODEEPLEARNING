### KAGGLE COMPETITIONS
#### HOUSE PRICES - ADVANCED REGRESSION TECHNIQUES

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques


```python
import torch
import torch.nn as nn
import pandas as pd
import itertools
import numpy as np
```


```python
# LOAD TRAINING DATASET AND TEST DATASET.

hp_training_set_pd = pd.read_csv("../data/train.csv");
hp_test_set_pd = pd.read_csv("../data/test.csv");

print("Training set size:", len(hp_training_set_pd));
print("Test set size:", len(hp_test_set_pd));
```

    Training set size: 1460
    Test set size: 1459



```python
# CONCATENATE THE DATASETS TO FACILITATE THE TRANSFORMATION ON BOTH SETS.

concat_sets = pd.concat((hp_training_set_pd.iloc[:,1:-1], hp_test_set_pd.iloc[:,1:]));

print(concat_sets.shape);
```

    (2919, 79)



```python
# TRANSFORM DATASETS'S FEATURES.

numeric_features = concat_sets.dtypes[concat_sets.dtypes != 'object'].index;

# Fill NaN entries using the average of their respective columns.

concat_sets[numeric_features] = concat_sets[numeric_features].apply(lambda x: x.fillna(x.mean()));

# Standardize each numeric feature.

concat_sets[numeric_features] = concat_sets[numeric_features].apply(lambda x: (x - x.mean()) / x.std());

# One-hot encoding for discrete features.

concat_sets = pd.get_dummies(concat_sets, dummy_na=True);

print(concat_sets.shape);
```

    (2919, 331)



```python
# CONVERT THE DATAFRAME INTO PYTORCH TENSOR AND DIVIDE IT INTO TRAINING SET AND TEST SET.

len_train_set = len(hp_training_set_pd);

hp_training_set = torch.tensor(concat_sets[:len_train_set].values, dtype=torch.float32);
hp_test_set = torch.tensor(concat_sets[len_train_set:].values, dtype=torch.float32);
hp_test_set_ids = torch.tensor(hp_test_set_pd.iloc[:,0].values);

train_labels = torch.tensor(hp_training_set_pd["SalePrice"].values, dtype=torch.float32);
```

***
### *MODEL*


```python
num_in_features = hp_training_set.shape[1];
num_hidden_units_h1 = 16;
num_hidden_units_h2 = 16;
num_hidden_units_h3 = 16;
num_out_label = 1;

num_hidden_units_h1_1 = 32;
num_hidden_units_h2_1 = 32;


neural_net_1 = nn.Sequential(nn.Linear(num_in_features, num_out_label));


neural_net_2 = nn.Sequential(
                nn.Linear(num_in_features, num_hidden_units_h1),
                nn.ReLU(),
                nn.Linear(num_hidden_units_h1, num_out_label));

# Test_error = 0.17443
neural_net_3 = nn.Sequential(
                nn.Linear(num_in_features, num_hidden_units_h1),
                nn.ReLU(),
                nn.Linear(num_hidden_units_h1, num_hidden_units_h2),
                nn.ReLU(),
                nn.Linear(num_hidden_units_h2, num_out_label));

# Test_error = 0.12957
neural_net_4 = nn.Sequential(
                nn.Linear(num_in_features, num_hidden_units_h1),
                nn.PReLU(),
                nn.Linear(num_hidden_units_h1, num_hidden_units_h2),
                nn.PReLU(),
                nn.Linear(num_hidden_units_h2, num_out_label));

# Test_error = 12.02373
# Training error = 0.1036 
# ! OVERFITTING !
neural_net_5 = nn.Sequential(
                nn.Linear(num_in_features, num_hidden_units_h1_1),
                nn.PReLU(),
                nn.Linear(num_hidden_units_h1_1, num_hidden_units_h2_1),
                nn.PReLU(),
                nn.Linear(num_hidden_units_h2_1, num_out_label));


neural_net_models = [neural_net_1, neural_net_2, neural_net_3];
```

***
### *INIT PARAMETERS*


```python
def initParameters(model):
    if isinstance(model, nn.Linear):
        init_param = 2 / np.sqrt(model.weight.size(1));
        model.weight.data.uniform_(-init_param, init_param);
        model.bias.data.uniform_(-init_param, init_param);
```

***
### *LOSS*


```python
mse_loss = nn.MSELoss();
```


```python
def rmseLoss(y_hat, y):
    
    for i in range(len(y_hat)):
        if y_hat[i] <= 0:
            y_hat[i] = 1/(-y_hat[i] + 1);
    
    y_hat, y = torch.log(y_hat), torch.log(y).reshape(y_hat.shape);

    return torch.sqrt(mse_loss(y_hat, y));
```


```python
learning_rates = torch.tensor([0.01,0.001]);
weights_decay = torch.tensor([0,3,6]);

hyperparameters_space = list(itertools.product(learning_rates, weights_decay));
```

***
## ONE-OFF SET TEST FROM TRAINING SET APPROACH

***
### *SELECTING AND TRAINING*
***

#### SELECTING PART


```python
len(hp_training_set)
```




    1460




```python
ooset_size = 146;

oo_test = hp_training_set[0:ooset_size];
oo_test_labels = train_labels[0:ooset_size];

oo_train = hp_training_set[ooset_size:];
oo_train_labels = train_labels[ooset_size:];
```


```python
def dataLoaderUtils(X, Y, batch_size):
    return torch.utils.data.DataLoader(
                    dataset=torch.utils.data.TensorDataset(X, Y),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2);
```


```python
def ooHyperParams(model, loss, hyperparameters_space, loaded_datasets, oo_train_set, oo_train_labels_set, 
       oo_test_set, oo_test_labels_set, batch_size, num_epochs):
    
    training_errors, testing_errors = [], []; 
    
    for i in range(len(hyperparameters_space)):
        
        optimizer = torch.optim.SGD(
                        model.parameters(), 
                        lr=hyperparameters_space[i][0], 
                        weight_decay=hyperparameters_space[i][1]);
        
        training_errors.append(ooTrain(model, loss, optimizer, loaded_datasets, 
                                     oo_train_set, oo_train_labels_set, num_epochs));
        testing_errors.append(ooTest(model, loss, oo_test_set, oo_test_labels_set));
        
        model.apply(initParameters);
        
    return training_errors, testing_errors;
```


```python
def ooTrain(model, loss, optimizer, loaded_datasets, oo_train_set, oo_train_labels_set, num_epochs):
    
    for epoch in range(num_epochs):
        for X,y in loaded_datasets:
            l = loss(model(X),y);
            l.backward();
            optimizer.step();
            optimizer.zero_grad();
            
    return loss(model(oo_train_set), oo_train_labels_set);
```


```python
def ooTest(model, loss, oo_test_set, oo_test_labels_set):
    return loss(model(oo_test_set), oo_test_labels_set);
```


```python
def ooModels(models, loss, hyperparameters_space, oo_train_set, oo_train_labels_set, 
       oo_test_set, oo_test_labels_set, batch_size, num_epochs, num_inits):
    
    training_errors = torch.empty((len(models), num_inits, len(hyperparameters_space)));
    testing_errors = torch.empty((len(models), num_inits, len(hyperparameters_space)));
    
    loaded_datasets = dataLoaderUtils(oo_train_set, oo_train_labels_set, batch_size);   
    
    for i in range(len(models)):
        model = models[i];
        
        print(f'model{i + 1}');
        
        for init in range(num_inits):
            tr_e, te_e = ooHyperParams(model, loss, hyperparameters_space, loaded_datasets, oo_train_set, 
                                       oo_train_labels_set, oo_test_set, oo_test_labels_set, batch_size, 
                                       num_epochs);
            
            training_errors[i,init] = torch.tensor(tr_e);
            testing_errors[i,init] = torch.tensor(te_e);
            
    return training_errors, testing_errors;
```


```python
training_errors, testing_errors = ooModels(neural_net_models, rmseLoss, hyperparameters_space, oo_train, 
                                           oo_train_labels, oo_test, oo_test_labels, batch_size=ooset_size, 
                                           num_epochs=50, num_inits=1);
```

    model1
    model2
    model3



```python
training_errors
```




    tensor([[[ 8.0057, 10.9046, 11.2509, 10.3827, 10.9354, 11.2525]],
    
            [[ 8.1660, 11.4539, 12.1465, 10.2421, 11.4410, 12.0987]],
    
            [[ 7.2735, 12.5819, 12.9292,  9.8731, 11.9166, 12.9038]]])




```python
testing_errors
```




    tensor([[[ 7.9779, 10.8767, 11.2229, 10.3535, 10.9073, 11.2245]],
    
            [[ 8.1368, 11.4258, 12.1183, 10.2138, 11.4128, 12.0696]],
    
            [[ 7.2465, 12.5531, 12.9005,  9.8495, 11.8832, 12.8756]]])



#### TRAINING PART ON THE CHOSEN MODEL


```python
def train(model, loss, optimizer, training_set, labels, batch_size, num_epochs):
    
    loaded_datasets = dataLoaderUtils(training_set, labels, batch_size);
    
    for epoch in range(num_epochs):
        for X,y in loaded_datasets:
            l = loss(model(X),y);
            l.backward();
            optimizer.step();
            optimizer.zero_grad();
        
#         print(f'epoch {epoch+1} ' f'training error {loss(model(training_set), labels)}');
        
    print(f'training error {loss(model(training_set), labels)}');
```


```python
optimizer_5 = torch.optim.SGD(neural_net_5.parameters(), lr=0.01, weight_decay=0);

train(neural_net_5, rmseLoss, optimizer_5, hp_training_set, train_labels, 292, 1000);
```

    training error 2.7864830493927


***
### *TESTING AND SUBMITTING PART*


```python
def testAndSubmit(model, test_set, test_set_pd):
    y_hat = model(test_set).detach().numpy();
    test_set_pd['SalePrice'] = pd.Series(y_hat.reshape(1,-1)[0]);
    submission_file = pd.concat([test_set_pd['Id'], test_set_pd['SalePrice']], axis=1);
    
    submission_file.to_csv("House_price_kaggle_sub.csv", index=False);
```


```python
testAndSubmit(neural_net_5, hp_test_set, hp_test_set_pd);
```

***
## *K-FOLD CROSS VALIDATION APPROACH*

***
### *K-FOLD CROSS VALIDATION OVER HYPERPARAMETERS'S OPTIMIZER SGD*


```python
# Training set size : 1460
# 1460 = 146 * 10 
# => k = 10;

def kFCV(k, model, loss, hyperparameters_space, training_set, labels, num_epochs):
    
    training_errors, testing_errors = [], [];
    
    for i in range(len(hyperparameters_space)):
        
        training_errors_i, testing_errors_i = [], [];
        
        optimizer = torch.optim.SGD(
                        model.parameters(), 
                        lr=hyperparameters_space[i][0], 
                        weight_decay=hyperparameters_space[i][1]);
        
        for j in range(k):
            Sj, training_set_minus_Sj = kFoldSetsj(k, j, training_set, labels);
            Sj = next(iter(Sj));
            Sj_features = Sj[0];
            Sj_labels = Sj[1];
            
            training_errors_i.append(trainkFCV(model, loss, optimizer, training_set_minus_Sj, 
                                              training_set, labels, num_epochs));
            
            testing_errors_i.append(loss(model(Sj_features), Sj_labels));
            
        
        training_errors.append(statistics.fmean(training_errors_i));
        testing_errors.append(statistics.fmean(testing_errors_i));
        
        initParameters(model);
        
    return training_errors, testing_errors;
```


```python
def kFoldSetsj(k, j, training_set, labels):
    
    Sj_size = math.ceil(len(training_set) / k);
    
    Sj_idx = torch.arange(j*Sj_size, min((j+1)*Sj_size, len(training_set)));
    
    Sj_features = training_set[Sj_idx];
    Sj_labels = labels[Sj_idx];
    
    training_set_minus_Sj_idx = torch.isin(torch.arange(0,len(training_set)), Sj_idx, invert=True);
    
    training_set_minus_Sj_features = training_set[training_set_minus_Sj_idx];
    training_set_minus_Sj_labels = labels[training_set_minus_Sj_idx]; 
        

    Sj = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.TensorDataset(Sj_features, Sj_labels),
                    batch_size=Sj_size,
                    num_workers=2);
    
    training_set_minus_Sj = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.TensorDataset(training_set_minus_Sj_features, 
                                                           training_set_minus_Sj_labels),
                    batch_size=Sj_size,
                    shuffle=True,
                    num_workers=2);
    
    return Sj, training_set_minus_Sj;
```

***
### *K-FOLD CROSS VALIDATION OVER MODELS*


```python
def kFCVmodels(k, models, loss, hyperparameters_space, training_set, labels, num_epochs, num_inits):
    
    training_errors = torch.empty((len(models), len(hyperparameters_space)));
    testing_errors = torch.empty((len(models), len(hyperparameters_space)));

    for i in range(len(models)):
        
        print(f'model{i + 1}');
        
        model = models[i];
        
        for init in range(num_inits):
            initParameters(model);
            
            tr_e, te_e = kFCV(k, model, loss, hyperparameters_space, training_set, labels, num_epochs);

            if init == 0:
                training_errors[i] = torch.tensor(tr_e);
                testing_errors[i] = torch.tensor(te_e);
            else:
                if statistics.fmean(tr_e) < statistics.fmean(training_errors[model]):
                    training_errors[i] = torch.tensor(tr_e);
                if statistics.fmean(te_e) < statistics.fmean(testing_errors[model]):
                    testing_errors[i] = torch.tensor(te_e);
        
    
    argmin_hyperparam = torch.argmin(testing_errors, dim=1);
    argmin_model = torch.argmin(testing_errors[range(len(models)), argmin_hyperparam]);
    
    return argmin_hyperparam, argmin_model, training_errors, testing_errors; 
```

***
### *TRAINING*


```python
def trainkFCV(model, loss, optimizer, training_set_minus_Sj, training_set, labels, num_epochs):
    for epoch in range(num_epochs):
        for X,y in training_set_minus_Sj:
            l = loss(model(X),y);
            l.backward();
            optimizer.step();
            optimizer.zero_grad();
            
    return loss(model(training_set), labels);
```
