import numpy as np 
import pandas as pd 
import pickle

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from foolbox import PyTorchModel
from foolbox.attacks import L2AdditiveGaussianNoiseAttack

torch.manual_seed(0)
np.random.seed(0)

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 20)
        self.layer3 = nn.Linear(20, 1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

dataframe = pd.read_csv('../Data/auto_filtered.csv', header = 0)
dataset = dataframe.values.astype(float)
Y = dataframe.loc[:,'mpg']
X = dataset[:,:-1]

Y_binned, bins = pd.cut(Y, bins = 10, labels = False, retbins = True)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

features_train, features_test, labels_train, labels_test = train_test_split(X, Y, random_state=42, shuffle=True)
x_train, y_train = Variable(torch.from_numpy(features_train)).float(), Variable(torch.from_numpy(labels_train.values)).float()
x_test, y_test =  Variable(torch.from_numpy(features_test)).float(), Variable(torch.from_numpy(labels_test.values)).float()
x_final, y_final =  Variable(torch.from_numpy(X)).float(), Variable(torch.from_numpy(Y.values)).float()

model = Model(features_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
loss_fn = nn.MSELoss()
epochs = 100

def print_(loss):
    print ("The loss calculated: ", loss)

print("Training")

for epoch in range(1, epochs+1):
    print ("Epoch #",epoch)
    y_pred = model.forward(x_train)
    loss = loss_fn(y_pred.squeeze(), y_train)
    print_(loss.item())
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step() 

print("Training error:", mean_squared_error(model(x_train).detach().numpy(),y_train))
y_pred = model(x_test).detach().numpy()
print("Test error:", mean_squared_error(y_pred, y_test))

fmodel = PyTorchModel(model.eval(), bounds=(-1, 2))
attack = L2AdditiveGaussianNoiseAttack()
epsilons = np.linspace(0.0, 6.0, num=6, endpoint=False)
advs, _, success = attack(fmodel, x_final, y_final, epsilons=epsilons)

perturbed = np.zeros((len(X),7))
added = np.zeros(len(X))

print("Finding adversarial inputs")

for eps, adv in zip(epsilons, advs):
    if 0 not in added:
        break
    pred = model(adv)
    pred = pred.detach().numpy().ravel()
    pred_binned = np.digitize(pred, bins)
    dif = [True if i[0]==i[1] else False for i in zip(pred_binned, Y_binned)] #True if class is correct
    for i in range(len(dif)):
        if dif[i]==False and added[i]==0:
            perturbed[i] = adv[i]
            added[i] = 1
    print(eps, mean_squared_error(Y, pred))

perturbed_output = scaler.inverse_transform(perturbed)
perturbed_output = [[max(0,round(x)) if i!=4 else max(0,round(x,1)) for i,x in enumerate(nested)] for nested in perturbed_output]

Y_output = Y.values
Y_output = np.reshape(Y_output, (len(Y_output), 1))

pred_final = model(Variable(torch.from_numpy(perturbed)).float())
pred_final = pred_final.detach().numpy().ravel()
pred_final = np.reshape(pred_final, (len(pred), 1))

pred_final_binned = np.digitize(pred_final, bins)

output = np.hstack((perturbed_output,Y_output,pred_final))

pd.DataFrame(output).to_csv("../Data/auto_perturbed.csv", index = False)

with open('../Outputs/auto_label_bins', 'wb') as fp:
    pickle.dump(bins, fp)

