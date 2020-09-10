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
model_type = str(sys.argv[3])

sal_path = data_type 

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

def get_saliency(model, loaderFull, classes):
    model.zero_grad()
    
    #model.eval()
    for param in model.parameters():
        param.requires_grad = False
    saliencies = []
    saliencies_wrong = []
    count = 0
    for i, data in enumerate(loaderFull):
        count+=1
        if i%1000 == 0:
            print(i)
        x, y = data
        x = x.to(device)
        y = y.to(device)
        x = x.permute(1,0,2)
        x.requires_grad_()
        _, output = model(x)


        grad_outputs = torch.zeros(x.shape[1], classes).to(device)
        grad_outputs[:, y] = 1

        #x.cpu()
        output.backward(gradient=grad_outputs)

        grads = x.grad

        saliency = np.squeeze(grads.cpu().numpy())
        saliencies.append(saliency)
    return saliencies, saliencies_wrong



X = np.load("./data/"+data_type+"/data.npy").astype(float)
Y = np.load("./data/"+data_type+"/labels.npy").astype(int)

X_full = torch.from_numpy(X).float()
Y_full = torch.from_numpy(Y).long()

dataLoaderFull = get_data_loader(X_full, Y_full, 1)

classes = int(np.max(Y)+1)
model = LSTM(X.shape[2], 100, X.shape[1], 100, int(np.max(Y)+1)).float()

model.load_state_dict(torch.load("./models/"+model_type+"/"+sal_path+"/model_"+str(run)+".pt"))
model.cuda()
saliencies, saliencies_wrong = get_saliency(model, dataLoaderFull, classes)

saliencies = np.stack(saliencies, axis=0)
print(saliencies.shape)
if not os.path.exists("./saliencies/"+model_type+"/"+sal_path):
    os.mkdir("./saliencies/"+model_type+"/"+sal_path)
np.save("./saliencies/"+model_type+"/"+sal_path+"/saliencies_"+str(run), saliencies)