import numpy as np 
import pandas as pd 
import sys

from collections import defaultdict

import copy
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from foolbox import PyTorchModel
from foolbox.attacks import L2AdditiveGaussianNoiseAttack

## Assume no missing values

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

le_labels = LabelEncoder()
labels_encoded = le_labels.fit_transform(labels)

d = defaultdict(LabelEncoder)

le_wc = LabelEncoder()
le_wc.fit(X[['workclass']])
le_ed = LabelEncoder()
le_ed.fit(X[['education']])
le_ms = LabelEncoder()
le_ms.fit(X[['marital-status']])
le_oc = LabelEncoder()
le_oc.fit(X[['occupation']])
le_re = LabelEncoder()
le_re.fit(X[['relationship']])
le_ra = LabelEncoder()
le_ra.fit(X[['race']])
le_se = LabelEncoder()
le_se.fit(X[['sex']])
le_nc = LabelEncoder()
le_nc.fit(X[['native-country']])

X_scaled = X_encoded

scaler = MinMaxScaler()
scaler.fit(X_scaled.iloc[:,0:6])
X_scaled.iloc[:,0:6] = scaler.transform(X_scaled.iloc[:,0:6])

X_final = X_scaled.iloc[0:1000,:]
y_final = labels_encoded[0:1000]

X_final, y_final = Variable(torch.from_numpy(X_final.values)).float(), Variable(torch.from_numpy(y_final)).long()

X_final[:,6:13] = F.softmax(X_final[:,6:13], dim = -1)
X_final[:,13:29] = F.softmax(X_final[:,13:29], dim = -1)
X_final[:,29:36] = F.softmax(X_final[:,29:36], dim = -1)
X_final[:,36:50] = F.softmax(X_final[:,36:50], dim = -1)
X_final[:,50:56] = F.softmax(X_final[:,50:56], dim = -1)
X_final[:,56:61] = F.softmax(X_final[:,56:61], dim = -1)
X_final[:,61:63] = F.softmax(X_final[:,61:63], dim = -1)
X_final[:,63:104] = F.softmax(X_final[:,63:104], dim = -1)

features_train, features_test, labels_train, labels_test = train_test_split(X_scaled, labels_encoded, random_state=42, shuffle=True)
x_train, y_train = Variable(torch.from_numpy(features_train.values)).float(), Variable(torch.from_numpy(labels_train)).long()

model = Model(features_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
epochs = 100

"""
Format after one-hot encoding: (column ranges inclusive)
    0 : age
    1 : fnlwgt
    2 : education-num
    3 : capital-gain
    4 : capital-loss
    5 : hours-per-week
    6-12 : workclass (7) -- 8?
    13-28 : education (16)
    29-35 : marital status (7)
    36-49 : occupation(14)
    50-55 : relationship (6)
    56-60 : race (5)
    61-62 : sex (2)
    63-103 : native country (41)
"""

x_train[:,6:13] = F.softmax(x_train[:,6:13], dim = -1)
x_train[:,13:29] = F.softmax(x_train[:,13:29], dim = -1)
x_train[:,29:36] = F.softmax(x_train[:,29:36], dim = -1)
x_train[:,36:50] = F.softmax(x_train[:,36:50], dim = -1)
x_train[:,50:56] = F.softmax(x_train[:,50:56], dim = -1)
x_train[:,56:61] = F.softmax(x_train[:,56:61], dim = -1)
x_train[:,61:63] = F.softmax(x_train[:,61:63], dim = -1)
x_train[:,63:104] = F.softmax(x_train[:,63:104], dim = -1)

def print_(loss):
    print ("The loss calculated: ", loss)

for epoch in range(1, epochs+1):
    print ("Epoch #",epoch)
    y_pred = model.forward(x_train)
    loss = loss_fn(y_pred.squeeze(), y_train)
    # print_(loss.item())
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step() 

x_test, y_test =  Variable(torch.from_numpy(features_test.values)).float(), Variable(torch.from_numpy(labels_test)).long()
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
advs, _, success = attack(fmodel, X_final, y_final, epsilons=epsilons)

perturbed = np.zeros((len(X_final),104))
added = np.zeros(len(X_final))

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
    dif = [True if i[0]==i[1] else False for i in zip(pred, y_final)] #True if class is correct
    for i in range(len(dif)):
        if dif[i]==False and added[i]==0:
            perturbed[i] = adv[i]
            added[i] = 1
    #print(accuracy_score(model.predict(adv),y_test))

perturbed_tensor = Variable(torch.from_numpy(perturbed)).float()
pred = model.predict(perturbed_tensor)
# print(accuracy_score(pred,y_test))

# diff = pred == y_test
# print(len(y_test))
# print(np.count_nonzero(diff))

actual_class = le_labels.inverse_transform(y_final)
actual_class = np.reshape(actual_class, (len(actual_class), 1))
pred = pred.detach().numpy()
pred_class = le_labels.inverse_transform(pred)
pred_class = np.reshape(pred_class, (len(pred_class), 1))

np.set_printoptions(suppress=True, threshold=sys.maxsize)

perturbed_1 = scaler.inverse_transform(perturbed[:,0:6])
perturbed_1 = np.clip(np.rint(perturbed_1), a_min = 0, a_max = None)
perturbed_1a = perturbed_1[:,0]
perturbed_1a = np.reshape(perturbed_1a, (len(perturbed_1a), 1))
perturbed_1b = perturbed_1[:,1]
perturbed_1b = np.reshape(perturbed_1b, (len(perturbed_1b), 1))
perturbed_1c = perturbed_1[:,2]
perturbed_1c = np.reshape(perturbed_1c, (len(perturbed_1c), 1))
perturbed_1d = perturbed_1[:,3:6]

perturbed_2 = np.argmax(perturbed[:,6:13], axis = 1)
perturbed_2 = le_wc.inverse_transform(perturbed_2)
perturbed_2 = np.reshape(perturbed_2, (len(perturbed_2), 1))

perturbed_3 = np.argmax(perturbed[:,13:28], axis = 1)
perturbed_3 = le_ed.inverse_transform(perturbed_3)
perturbed_3 = np.reshape(perturbed_3, (len(perturbed_3), 1))

perturbed_4 = np.argmax(perturbed[:,29:36], axis = 1)
perturbed_4 = le_ms.inverse_transform(perturbed_4)
perturbed_4 = np.reshape(perturbed_4, (len(perturbed_4), 1))

perturbed_5 = np.argmax(perturbed[:,36:50], axis = 1)
perturbed_5 = le_oc.inverse_transform(perturbed_5)
perturbed_5 = np.reshape(perturbed_5, (len(perturbed_5), 1))

perturbed_6 = np.argmax(perturbed[:,50:56], axis = 1)
perturbed_6 = le_re.inverse_transform(perturbed_6)
perturbed_6 = np.reshape(perturbed_6, (len(perturbed_6), 1))

perturbed_7 = np.argmax(perturbed[:,56:61], axis = 1)
perturbed_7 = le_ra.inverse_transform(perturbed_7)
perturbed_7 = np.reshape(perturbed_7, (len(perturbed_7), 1))

perturbed_8 = np.argmax(perturbed[:,61:63], axis = 1)
perturbed_8 = le_se.inverse_transform(perturbed_8)
perturbed_8 = np.reshape(perturbed_8, (len(perturbed_8), 1))

perturbed_9 = np.argmax(perturbed[:,63:104], axis = 1)
perturbed_9 = le_nc.inverse_transform(perturbed_9)
perturbed_9 = np.reshape(perturbed_9, (len(perturbed_9), 1))

perturbed_final = np.hstack(
    (perturbed_1a, perturbed_2, perturbed_1b, perturbed_3, perturbed_1c, 
    perturbed_4, perturbed_5, perturbed_6, perturbed_7, perturbed_8, perturbed_1d, perturbed_9))

output = np.hstack((perturbed_final, actual_class, pred_class))

pd.DataFrame(output).to_csv("../Data/adult_perturbed.csv", index = False)