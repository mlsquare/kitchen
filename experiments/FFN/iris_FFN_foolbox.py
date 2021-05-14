import tensorflow
from tensorflow.keras import backend as K

import pandas as pd
import numpy as np
import eagerpy as ep

from sklearn.preprocessing import LabelEncoder
from foolbox import TensorFlowModel, accuracy
from foolbox.attacks import L2AdditiveGaussianNoiseAttack
from sklearn.metrics import accuracy_score

np.random.seed(0)
tensorflow.random.set_seed(0)

model = tensorflow.keras.models.load_model('iris_ffn.h5')

dataframe = pd.read_csv("../Data/iris_scaled.csv", header=0)
dataset = dataframe.values
X = dataset[:,:4].astype(float)
labels = dataset[:,4]

print(labels)

le = LabelEncoder()
Y = le.fit_transform(labels)

fmodel = TensorFlowModel(model, bounds=(-10, 10))
attack = L2AdditiveGaussianNoiseAttack()
epsilons = np.linspace(0.0, 6.0, num=480, endpoint=False)
advs, _, success = attack(fmodel, X, K.constant(Y), epsilons=epsilons)

perturbed = np.zeros((len(X),4))
added = np.zeros(len(X))

for eps, adv in zip(epsilons, advs):
    if 0 not in added:
        break
    pred = model.predict(adv)
    pred_class = np.argmax(pred, axis=1)
    dif = [True if i[0]==i[1] else False for i in zip(pred_class, Y)] #True if class is correct
    for i in range(len(dif)):
        if dif[i]==False and added[i]==0:
            perturbed[i] = adv[i]
            added[i] = 1
    #print(eps, accuracy_score(Y, pred_class))

pred = model.predict(perturbed)
pred_class = np.argmax(pred, axis=1)
labels = np.reshape(labels, (len(labels), 1))
pred_class = le.inverse_transform(pred_class)
pred_class = np.reshape(pred_class, (len(pred_class), 1))
op = np.hstack((perturbed,labels,pred_class))
print(op)

pd.DataFrame(op).to_csv("../Data/iris_perturbed.csv", index = False) 
