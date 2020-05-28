import numpy as np #y
import pandas as pd #y

import copy
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

import eagerpy as ep

from foolbox import PyTorchModel, accuracy
from foolbox.attacks import L2AdditiveGaussianNoiseAttack

torch.manual_seed(0)
np.random.seed(0)

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 20)
        self.layer3 = nn.Linear(20, 2)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    def predict(self,x): 
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

dataset = pd.read_csv('../Data/adult_filtered.csv', header = 0)
labels = dataset['target']
X = dataset.drop(columns=['target'])

X_encoded = pd.get_dummies(X, dummy_na=False)

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

scaler = MinMaxScaler()
scaler.fit(X_encoded)
X_scaled = scaler.transform(X_encoded)

## Assume no missing values

# X_scaled[:,6:13] = F.softmax(X_scaled[:,6:13], dim = -1)
# X_scaled[:,13:29] = F.softmax(X_scaled[:,13:29], dim = -1)
# X_scaled[:,29:36] = F.softmax(X_scaled[:,29:36], dim = -1)
# X_scaled[:,36:50] = F.softmax(X_scaled[:,36:50], dim = -1)
# X_scaled[:,50:56] = F.softmax(X_scaled[:,50:56], dim = -1)
# X_scaled[:,56:61] = F.softmax(X_scaled[:,56:61], dim = -1)
# X_scaled[:,61:63] = F.softmax(X_scaled[:,61:63], dim = -1)
# X_scaled[:,63:104] = F.softmax(X_scaled[:,63:104], dim = -1)

features_train, features_test, labels_train, labels_test = train_test_split(X_scaled, labels_encoded, random_state=42, shuffle=True)
x_train, y_train = Variable(torch.from_numpy(features_train)).float(), Variable(torch.from_numpy(labels_train)).long()

model = Model(features_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
epochs = 100

def print_(loss):
    print ("The loss calculated: ", loss)

"""
Format after one-hot encoding: (column ranges inclusive)
    0 : age
    1 : fnlwgt
    2 : education-num
    3 : capital-gain
    4 : capital-loss
    5 : hours-per-week
    6-12 : workclass
    13-28 : education
    29-35 : marital status
    36-49 : occupation
    50-55 : relationship
    56-60 : race
    61-62 : sex
    63-103 : native country
"""

x_train[:,6:13] = F.softmax(x_train[:,6:13], dim = -1)
x_train[:,13:29] = F.softmax(x_train[:,13:29], dim = -1)
x_train[:,29:36] = F.softmax(x_train[:,29:36], dim = -1)
x_train[:,36:50] = F.softmax(x_train[:,36:50], dim = -1)
x_train[:,50:56] = F.softmax(x_train[:,50:56], dim = -1)
x_train[:,56:61] = F.softmax(x_train[:,56:61], dim = -1)
x_train[:,61:63] = F.softmax(x_train[:,61:63], dim = -1)
x_train[:,63:104] = F.softmax(x_train[:,63:104], dim = -1)

for epoch in range(1, epochs+1):
    print ("Epoch #",epoch)
    y_pred = model.forward(x_train)
    loss = loss_fn(y_pred.squeeze(), y_train)
    # print_(loss.item())
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step() 

x_test, y_test =  Variable(torch.from_numpy(features_test)).float(), Variable(torch.from_numpy(labels_test)).long()
x_softmax = copy.deepcopy(x_test)
x_softmax[:,6:13] = F.softmax(x_softmax[:,6:13], dim = -1)
x_softmax[:,13:29] = F.softmax(x_softmax[:,13:29], dim = -1)
x_softmax[:,29:36] = F.softmax(x_softmax[:,29:36], dim = -1)
x_softmax[:,36:50] = F.softmax(x_softmax[:,36:50], dim = -1)
x_softmax[:,50:56] = F.softmax(x_softmax[:,50:56], dim = -1)
x_softmax[:,56:61] = F.softmax(x_softmax[:,56:61], dim = -1)
x_softmax[:,61:63] = F.softmax(x_softmax[:,61:63], dim = -1)
x_softmax[:,63:104] = F.softmax(x_softmax[:,63:104], dim = -1)
# pred = model(x_softmax)

print(accuracy_score(model.predict(x_softmax),y_test))

fmodel = PyTorchModel(model.eval(), bounds=(-1, 2))
attack = L2AdditiveGaussianNoiseAttack()
epsilons = np.linspace(0.0, 30.0, num=30, endpoint=False)
advs, _, success = attack(fmodel, x_test, y_test, epsilons=epsilons)

perturbed = np.zeros((len(x_test),104))
added = np.zeros(len(x_test))

print("Adversarial inputs")
for eps, adv in zip(epsilons, advs):
    if 0 not in added:
        break
    print(eps)
    adv[:,6:13] = F.softmax(adv[:,6:13], dim = -1)
    adv[:,13:29] = F.softmax(adv[:,13:29], dim = -1)
    adv[:,29:36] = F.softmax(adv[:,29:36], dim = -1)
    adv[:,36:50] = F.softmax(adv[:,36:50], dim = -1)
    adv[:,50:56] = F.softmax(adv[:,50:56], dim = -1)
    adv[:,56:61] = F.softmax(adv[:,56:61], dim = -1)
    adv[:,61:63] = F.softmax(adv[:,61:63], dim = -1)
    adv[:,63:104] = F.softmax(adv[:,63:104], dim = -1)
    pred = model.predict(adv)
    dif = [True if i[0]==i[1] else False for i in zip(pred, y_test)] #True if class is correct
    for i in range(len(dif)):
        if dif[i]==False and added[i]==0:
            perturbed[i] = adv[i]
            added[i] = 1
    #print(accuracy_score(model.predict(adv),y_test))

perturbed_tensor = Variable(torch.from_numpy(perturbed)).float()
pred = model.predict(perturbed_tensor)
print(accuracy_score(pred,y_test))

# diff = pred == y_test
# print(len(y_test))
# print(np.count_nonzero(diff))

actual_class = np.reshape(labels_test, (len(labels_test), 1))
pred = pred.detach().numpy()
pred_class = np.reshape(pred, (len(pred), 1))

output = np.hstack((perturbed, actual_class, pred_class))
print(output)

pd.DataFrame(output).to_csv("../Data/adult_perturbed.csv")