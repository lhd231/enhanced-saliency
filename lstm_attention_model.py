import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable

import sys
import os

run = int(sys.argv[1])
data_type = str(sys.argv[2])
train_size = float(sys.argv[3])




output_dir = "wtest_"+data_type


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(run)

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_nodes, sequence_size, fc_size, output_size):
        super(LSTM, self).__init__()
        self.sequence_size = sequence_size
        self.hidden = hidden_nodes
        self.lstm = nn.LSTM(input_size, hidden_nodes, bidirectional = True)

        self.fc_attn1 = torch.nn.Linear(hidden_nodes*2,50)
        self.fc_attn2 = torch.nn.Linear(50,1)

        self.fc_out = torch.nn.Linear(hidden_nodes*2,output_size)
        #self.fc_attn2 = torch.nn.Linear(fc_size, output_size)
        

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)

        lstm_attn = lstm_out.view([-1,self.hidden*2])
        lstm_attn = self.fc_attn2(self.fc_attn1(lstm_attn))
        lstm_attn = lstm_attn.view([self.sequence_size,-1])
        smax = torch.nn.Softmax(dim=0)
        lstm_attn = smax(lstm_attn)

        lstm_attn = lstm_attn.unsqueeze(2).expand_as(lstm_out) * lstm_out
        lstm_attn = torch.squeeze(torch.sum(lstm_attn, dim=0))

        lstm_out = self.fc_out(lstm_attn)

        smax = torch.nn.Softmax()
        lstm_out_smax = smax(lstm_out)
        return lstm_out, lstm_out_smax

class DataWithLabels(Dataset):
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
  def __len__(self): return len(self.X)
  def __getitem__(self, i): return self.X[i], self.Y[i]


def get_data_loader(X, Y, batch_size):
    dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size = batch_size)

    return dataLoader


def train_model(model, loader_train, loader_test, batch_size, epochs, learning_rate):

    loss = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.cuda()

    for epoch in range(epochs):
        accuracy_train_total = 0
        count_train = 0
        for i, data in enumerate(loader_train):
            x, y = data
            count_train += x.shape[0]
            x = x.permute(1,0,2)
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(x)
            l = loss(outputs, y)
            accuracy_train_total += l.sum().item()
            _, preds = torch.max(outputs.data, 1)
            accuracy = (preds == y).sum().item()
            accuracy_train_total += accuracy
            
            l.backward()

            optimizer.step()
        accuracy_total = 0
        count = 0
        for i, data in enumerate(loader_test):
            x_test, y_test = data#next(iter(loader_test))
            count += x_test.shape[0]
            x_test = x_test.permute(1,0,2)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            outputs, _ = model(x_test)
            l = loss(outputs, y_test)
            _, preds = torch.max(outputs.data, 1)
            accuracy = (preds == y_test).sum().item()
            accuracy_total += accuracy

        print(count)
        print(accuracy_total)
        print(count_train)
        print(accuracy_train_total/count_train)
        test_acc = accuracy_total/count
        print("epoch: "+str(epoch)+", train acc: "+str(test_acc))
        if test_acc == 1:
            break
    return optimizer

def get_saliency(model, loaderFull, optimizer):
    model.zero_grad()
    #model.eval()
    for param in model.parameters():
        param.requires_grad = False
    saliencies = []
    count = 0
    for i, data in enumerate(loaderFull):
        count+=1

        print(i)
        x, y = data
        x = x.to(device)
        y = y.to(device)
        x = x.permute(1,0,2)
        x.requires_grad_()
        _, output = model(x)

        grad_outputs = torch.zeros(x.shape[1], 3).to(device)
        grad_outputs[:, y] = 1

        #x.cpu()
        output.backward(gradient=grad_outputs)

        grads = x.grad

        saliency = np.squeeze(grads.cpu().numpy())
        saliencies.append(saliency)
    return saliencies



print(torch.cuda.is_available())

X = np.load("./data/"+data_type+"/data.npy").astype(float)
Y = np.load("./data/"+data_type+"/labels.npy").astype(int)

if len(X.shape) == 2:
    X = np.reshape(X,[X.shape[0],X.shape[1],1])

print(X.shape)
train_end = int((train_size)*X.shape[0])
#trend_noise_3
#X = np.transpose(X,[0,2,1])


X_test = X[train_end:,:]
Y_test = Y[train_end:]

X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test).long()
X_train = X[:train_end,:]
Y_train = Y[:train_end]
print(X_train.shape)
print(X_test.shape)
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).long()



X_full = torch.from_numpy(X).float()
Y_full = torch.from_numpy(Y).long()

dataLoaderTrain = get_data_loader(X_train, Y_train, 200)
dataLoaderTest = get_data_loader(X_test, Y_test, 50)
model = LSTM(X.shape[2], 100, X.shape[1], 100, int(np.max(Y)+1)).float()

optimizer = train_model(model, dataLoaderTrain, dataLoaderTest, 1000, 150, .001)

if not os.path.exists("./models/rseam/"+output_dir):
    os.mkdir("./models/rseam/"+output_dir)
torch.save(model.state_dict(), "./models/rseam/"+output_dir+"/model_"+str(run)+".pt")

'''
dataLoaderFull = get_data_loader(X_full, Y_full, 1)

saliencies = get_saliency(model, dataLoaderFull, optimizer)


saliencies = np.stack(saliencies, axis=0)
print(saliencies.shape)
if not os.path.exists("./saliencies/rseam/"+data_type):
    os.mkdir("./saliencies/rseam/"+data_type)
np.save("./saliencies/rseam/"+data_type+"/saliencies_"+str(run), saliencies)
#np.save("./saliencies/1dcnn/"+data_type+"/permutation_"+str(run), perm)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''