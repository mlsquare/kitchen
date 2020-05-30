import tensorflow
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import eagerpy as ep
from foolbox import TensorFlowModel, accuracy
from foolbox.attacks import L2AdditiveGaussianNoiseAttack
from sklearn.metrics import accuracy_score

np.random.seed(0)
tensorflow.random.set_seed(0)

model = tensorflow.keras.models.load_model('iris_ffn.h5')

dataframe = pd.read_csv("../Data/iris_with_paths.csv", header=None)
dataset = dataframe.values
X = dataset[1:,1:5].astype(float)
Y = dataset[1:,5].astype(int)

fmodel = TensorFlowModel(model, bounds=(-10, 10))
attack = L2AdditiveGaussianNoiseAttack()
epsilons = np.linspace(0.0, 6.0, num=480, endpoint=False)
advs, _, success = attack(fmodel, X, K.constant(Y), epsilons=epsilons)

final_ip = np.zeros((len(X),4))
ip_added = np.zeros(len(X))

for eps, adv in zip(epsilons, advs):
    if 0 not in ip_added:
        break
    pred = model.predict(adv)
    pred_class = np.argmax(pred, axis=1)
    dif = [True if i[0]==i[1] else False for i in zip(pred_class, Y)] #True if class is correct
    for i in range(len(dif)):
        if dif[i]==False and ip_added[i]==0:
            final_ip[i] = adv[i]
            ip_added[i] = 1
    #print(dif)
    #print(eps, accuracy_score(Y, pred_class))

#print(final_ip)
#print(ip_added)

pred = model.predict(final_ip)
pred_class = np.argmax(pred, axis=1)
#print(pred_class)
Y = np.reshape(Y, (len(Y), 1))
pred_class = np.reshape(pred_class, (len(pred_class), 1))
op = np.hstack((final_ip,Y,pred_class))
print(op)

pd.DataFrame(op).to_csv("../Data/iris_perturbed.csv", index = False) # Change R file
