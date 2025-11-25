# Code for creating a spiral dataset from CS231n
import numpy as np

import matplotlib.pyplot as plt
N = 300 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(2*j,(2*j+2),N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
plt.show()

# Import torch
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup random seed
RANDOM_SEED = 42

# Create a dataset with Scikit-Learn's make_moons()
from sklearn.datasets import make_moons

X_,Y = make_moons(n_samples = 1000, noise = 0.07 , random_state = 42)
print(f'the first ten element:{X_[:10]}{Y[:10]}')
# a kind of numpy array

# Turn data into a DataFrame
import pandas as pd
df = pd.DataFrame(X_,columns = ['x1','x2'])
df['label'] = Y
# another way moons = pd.DataFrame({"x1":X[:,0] ,"x2":X[:,1],'label':y})
df.head(10)
print(df.label.value_counts())

# Visualize the data on a scatter plot
import matplotlib.pyplot as plt
plt.scatter(x=X_[:,0],y= X_[:,1],c = Y, cmap = plt.cm.RdYlBu)

# Turn data into tensors of dtype float

X = torch.from_numpy(X_).type(torch.float)
y = torch.from_numpy(Y).type(torch.float)
# Split the data into train and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split

print(X,Y)
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
print(X_train.size())
print(y_train.dtype)

import torch
from torch import nn

# Inherit from nn.Module to make a model capable of fitting the mooon data
class MoonModelV0(nn.Module):
    ## Your code here ##
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2,10)
        self.layer_2 = nn.Linear(10,10)
        self.layer_3 = nn.Linear(10,10)
        self.layer_4 = nn.Linear(10,1)
        self.relu = nn.ReLU()


    def forward(self, x):
        ## Your code here ##
        return self.relu(self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))))))

# Instantiate the model
## Your code here ##
model = MoonModelV0()

# Setup loss function
lossfn = nn.BCEWithLogitsLoss()
# Setup optimizer to optimize model's parameters
opt = torch.optim.SGD(params = model.parameters(),lr = 0.1)


# What's coming out of our model?
print('before training:')
# logits (raw outputs of model)
print("Logits:")
## Your code here ##
# transform to float32 is neccessary
print("y_logits",model(X_test)[:10])

# Prediction probabilities
print("Pred probs:")
## Your code here ##
print(torch.sigmoid(model(X_test)[:10]).squeeze())

# Prediction labels
print("Pred labels:")

labels = torch.round(torch.sigmoid(model(X_test)[:10]))
print(labels.dtype)
## Your code here ##

# Let's calculuate the accuracy using accuracy from TorchMetrics
#!pip -q install torchmetrics # Colab doesn't come with torchmetrics
from torchmetrics import Accuracy

## TODO: Uncomment this code to use the Accuracy function
#acc_fn = Accuracy(task="multiclass", num_classes=2).to(device) # send accuracy function to device
#acc_fn

def acc_fn(y_truth, y_pred):
    correct = torch.eq(y_truth,y_pred).sum().item()
    acc = (correct/ len(y_truth))*100
    return acc

y_train.size()


## TODO: Uncomment this to set the seed
torch.manual_seed(RANDOM_SEED)


# Setup epochs
epochs = 5000

# Send data to the device
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
model = model.to(device)

# Loop through the data
for epoch in range(epochs):
  ### Training
    model.train()
  # 1. Forward pass (logits output)
    y_logits = model(X_train).squeeze()
  # Turn logits into prediction probabilities
    y_prob = torch.sigmoid(y_logits)

  # Turn prediction probabilities into prediction labels
    y_pred = torch.round(y_prob)

  # 2. Calculaute the loss
    loss = lossfn(y_logits, y_train) # loss = compare model raw outputs to desired model outputs

  # Calculate the accuracy
    acc = acc_fn(y_train,y_pred) # the accuracy function needs to compare pred labels (not logits) with actual labels

  # 3. Zero the gradients
    opt.zero_grad()

  # 4. Loss backward (perform backpropagation)
    loss.backward()

  # 5. Step the optimizer (gradient descent)
    opt.step()

  ### Testing
    model.eval()
    with torch.inference_mode():
    # 1. Forward pass (to get the logits)
        y_logits_eval = model(X_test)
    # Turn the test logits into prediction labels
        y_pred_eval = torch.round(torch.sigmoid(y_logits_eval)).squeeze()

    # 2. Caculate the test loss/acc
        acc_test = acc_fn(y_test,y_pred_eval)

  # Print out what's happening every 100 epochs
    if epoch % 100 == 0:
        print(f'In Epoch:{epoch} the training loss{loss:3f},accuracy:{acc:.2f}|the testing accuracy:{acc_test:2f}')


#print(y_train[:10])
#print(y_pred[:10])
# I am continually confused on the set of hyperparameter.
# They seem like blackbox to me. I tune them but I don't have any confidence in the process.

# Plot the model predictions
import numpy as np

def plot_decision_boundary(model, X, y):
  
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")


    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), 
                         np.linspace(y_min, y_max, 101))

    # Make features
    coords = np.stack([xx, yy], axis = -1)
    coords = coords.reshape(-1,2)
    X_to_pred_on = torch.from_numpy(coords).type(torch.float)



    #X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # mutli-class
    else: 
        y_pred = torch.round(torch.sigmoid(y_logits)).squeeze() # binary
    
    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7) # both xx,yy and y_pred are in 101*101 form
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

# Plot decision boundaries for training and test sets
plot_decision_boundary(model,X,y)

# Create a straight line tensor
l = torch.linspace(-10.0,10.0,steps = 201) #steps is actually like numpy , the number of point you want to split


# Code for creating a spiral dataset from CS231n
import numpy as np
import matplotlib.pyplot as plt
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 1000 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.show()

# Turn data into tensors
import torch
X = torch.from_numpy(X).type(torch.float) # features as float32
y = torch.from_numpy(y).type(torch.LongTensor) # labels need to be of type long

# Create train and test splits
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state = 42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Let's calculuate the accuracy for when we fit our model
#!pip -q install torchmetrics # colab doesn't come with torchmetrics
#from torchmetrics import Accuracy

## TODO: uncomment the two lines below to send the accuracy function to the devic



# Prepare device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create model by subclassing nn.Module
class ClassifyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.Tanh(),
            nn.Linear(10,3),
        )

    def forward(self,x):
        return self.layers(x)

# Instantiate model and send it to device
model_1 = ClassifyModel()

# Setup data to be device agnostic
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
model_1 = model_1.to(device)

# Print out first 10 untrained model outputs (forward pass)
print("Logits:")
## Your code here ##
print(model_1(X_train[:10]))
print("Pred probs:")
## Your code here ##
print(torch.softmax(model_1(X_train[:10]),dim =1))### remember dim is mandatory for softmax as you must know which dim to operate on
print("Pred labels:")
print(torch.argmax(torch.softmax(model_1(X_train[:10]),dim = 1)[0,:]))

print('sum of each row:')
print((torch.softmax(model_1(X_train[:10]),dim = 1)[0,:]).sum())
foo = torch.softmax(model_1(X_train[:10]),dim = 1)
## Your code here ##
print('foo=',foo)
print(foo.argmax(dim = 1))
print(y_train[:10])

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_1.parameters(),lr = 0.1)

# Build a training loop for the model

# Loop over data
epochs = 5000


for epoch in range(epochs):

  ## Training
    model_1.train()
  # 1. Forward pass
    y_logits = model_1(X_train)
    y_prob = torch.softmax(y_logits, dim =1)
    y_pred = y_prob.argmax(dim = 1)
  # 2. Calculate the loss
    loss = loss_fn(y_logits,y_train)
  
  # 3. Optimizer zero grad
    optimizer.zero_grad()

  # 4. Loss backward
    loss.backward()

  # 5. Optimizer step
    optimizer.step()

  ## Testing
    model_1.eval()
    with torch.inference_mode() :

        # 1. Forward pass
        y_logits_eval = model_1(X_test)
        y_prob_eval = torch.softmax(y_logits_eval, dim =1)
        y_pred_test = y_prob_eval.argmax(dim = 1)
        # 2. Caculate loss and acc
        acc = acc_fn(y_test,y_pred_test)
        acc = acc_fn(y_train,y_pred)

  # Print out what's happening every 100 epochs
    if epoch %100 == 0:

        print(f"Epoch:{epoch},the training loss:{loss:2f}, training accuracy:{acc:2f}%,the testing accuracy:{acc:2f}%")

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)


