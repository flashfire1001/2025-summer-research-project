import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

#getting a dataset
torch.manual_seed(42)


train_data = datasets.MNIST(
    root = "data",
    train= True,
    download = False,
    transform= ToTensor(), # image comes as PIL format we want to turn into torch tensors
    target_transform=None,
)

test_data = datasets.MNIST(
    root = "data",
    train= False,
    download = False,
    transform= ToTensor(), # image comes as PIL format we want to turn into torch tensors
    target_transform=None,
)


image, label = train_data[1]
print(image.shape, label)
img = train_data.data[0]
print(img.shape) #img is of 0-255
print(image.squeeze()*255-img)

class_names = train_data.classes
print(len(class_names),
      class_names) # a python list

plt.imshow(image.squeeze(),cmap ='grey')
plt.title('digits')
plt.xlabel('pixels')
plt.ylabel('pixels')
plt.show()#plt prefer 0-1 image .

#A subplot is a small plot inside a larger figure window. This allows you to visualize multiple plots side-by-side or in a grid.

# then I will create a bunch of it
fig, axes = plt.subplots(3,3,figsize = (9,9))

for (i,ax) in enumerate(axes.flatten()):
    j = torch.randint(0,1000,[1]).item()
    image, label = train_data[j]
    ax.set_title(f'{i+1}th graph:{class_names[label]}')
    ax.imshow(image.squeeze(),cmap = 'grey')
    ax.plot()

# prepare dataloader

from torch.utils.data import DataLoader

batchsize = 32
train_dataloader = DataLoader(train_data, # turn data into iterable
                              batch_size = batchsize,
                              shuffle = True
                        )
test_dataloader = DataLoader(test_data,
                             batch_size = batchsize,
                             shuffle =False)

print(f"Dataloaders:{len(test_dataloader)*32},{len(train_dataloader)*32}") #just for verify, see if all the data is loaded

train_features_batch , train_labels_batch = next(iter(train_dataloader))
# I will have a check on this
print(train_features_batch.shape)
image = train_features_batch[0].squeeze()
label = train_labels_batch[0]
plt.imshow(image,cmap = 'grey')# why it is not zero( the real first one)? you give a shuffle!
plt.title(f"{class_names[label]}")




class MNISTClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 1,
                      out_channels = 10,
                      kernel_size = 3,
                      padding = 1,
                      stride = 1),

            nn.ReLU(), # add this for a pixel's grey value is positive
            nn.Conv2d(in_channels = 10,
                      out_channels = 10,
                      kernel_size = 3,
                      padding = 1,
                      stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2)

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 10,
                      out_channels = 10,
                      kernel_size = 3,
                      padding = 1,
                      stride = 1),

            nn.ReLU(), # add this for a pixel's grey value is positive
            nn.Conv2d(in_channels = 10,
                      out_channels = 10,
                      kernel_size = 3,
                      padding = 1,
                      stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2)

        )
        self.outputlayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
            in_features = 10 * 7 * 7,
            out_features = 10
        )
        )
        #output logits for further CrossEntropyLoss

    def forward(self,x):
        """

        :param x: a batch of image of size (bs, 1, 28, 28)
        :return: a prob vector of size (bs,10)
        """
        processed_features = self.block2(self.block1(x))

        return self.outputlayer(processed_features)

#create a instance
model_CL = MNISTClassifierModel()



# define functions for training and testing and evaluation
def accuracy_fn(y_pred,y_truth):
    """a function to calculate the accuracy, using the """
    correct = torch.eq(y_pred,y_truth).sum().item()
    total = y_truth.size(0)
    return correct / total * 100


def train_step(data_loader:DataLoader, model, optimizer, device):
    """the function cover all the codes for a single training epoch"""

    model = model.to(device)
    total_loss = 0
    total_acc = 0
    model.train()


    for i, (train_features_batch, train_labels_batch) in enumerate(data_loader):

        #sent the data to device
        train_features_batch = train_features_batch.to(device)
        train_labels_batch = train_labels_batch.to(device)

        y_logits = model(train_features_batch)

        y_prob = torch.softmax(y_logits,dim = 1)

        y_pred = y_prob.argmax(dim = 1)

        acc = accuracy_fn(y_pred, train_labels_batch)

        optimizer.zero_grad()

        loss = loss_fn(y_logits, train_labels_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss
        total_acc+= acc

    train_loss = total_loss / len(data_loader)
    train_acc = total_acc /len(data_loader)

    print(f"Train loss:{train_loss:.5f} | Train_accuracy:{train_acc:.3f}%")


def test_step(data_loader:DataLoader, model, device):
    """the function cover all the codes for a single test epoch"""

    model = model.to(device)
    total_loss = 0
    total_acc = 0
    model.eval()


    with torch.inference_mode():
        for i, (test_features_batch, test_labels_batch) in enumerate(data_loader):

            #sent the data to device
            test_features_batch = test_features_batch.to(device)
            test_labels_batch = test_labels_batch.to(device)

            y_logits = model(test_features_batch)

            y_prob = torch.softmax(y_logits,dim = 1)

            y_pred = y_prob.argmax(dim = 1)

            acc = accuracy_fn(y_pred, test_labels_batch)

            loss = loss_fn(y_logits, test_labels_batch)

            total_loss += loss.item()
            total_acc += acc

    test_loss = total_loss / len(data_loader)
    test_acc = total_acc /len(data_loader)

    print(f"Test loss:{test_loss:.5f} | Test accuracy:{test_acc:.3f}%")

# set up the loss function & the optimizer & device
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(params=model_CL.parameters(),lr = 0.1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'


epochs = 3
from tqdm import tqdm
for epoch in range(epochs):
    print(f"Epoch:{epoch} ----")

    train_step(train_dataloader,model_CL,opt,device)

    test_step(test_dataloader,model_CL,device)

# for i, (test_features_batch, test_labels_batch) in enumerate(train_dataloader):
#     if i == 1:
#         print(test_features_batch.shape)
#         model_CL(test_features_batch)
#
# then I will create a bunch of it
fig, axes = plt.subplots(3,3,figsize = (9,9))


for (i,ax) in enumerate(axes.flatten()):
    j = torch.randint(0,1000,[1]).item()
    image, label = train_data[j]
    image = image.unsqueeze(dim = 0)
    model_CL.to('cpu')
    model_CL.eval()
    y_logits = model_CL(image)

    y_prob = torch.softmax(y_logits,dim = 1)

    label_pred = y_prob.argmax(dim = 1)
    ax.set_title(f'{i+1}th graph:{class_names[label_pred]}')
    ax.imshow(image.squeeze(),cmap = 'grey')
    ax.plot()






def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)





