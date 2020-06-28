import tensorflow as tf
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from tensorflow.keras.layers import Dense, Input, concatenate, GRU, LSTM
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import jaccard_score
import re


'''
Load and preprocess dataset
Dataset available at https://github.com/mlsquare/kitchen/blob/dert/data/raw/adult.data.csv
'''
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
         'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
         'hours-per-week', 'native-country', 'target']

data = pd.read_csv('../input/dert-adult-dataset/adult.data.csv', delimiter=",", header=None, names=names)

data = data[data["workclass"] != " ?"]
data = data[data["occupation"] != " ?"]
data = data[data["native-country"] != " ?"]

# Convert categorical fields #
categorical_col = ['workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'native-country', 'target']

# categorical_col = ['target']
    
# for col in categorical_col:
#     categories = unique_of(data.col)
#     num_cat = count(categories)
#     for cat in categories:
#         data.col[cat] = index_of(cat in categories)

for col in categorical_col:
    b, c = np.unique(data[col], return_inverse=True)
    data[col] = c

feature_list = names[:14]
# Test train split #
X = data.loc[:, feature_list]
Y = data[['target']]

# Split the dataset into test and train datasets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)

df = pd.DataFrame(X)
df = df.reset_index().drop(columns='index')


# Standardizing continuous variables
cont_var = []

for i in list(df.columns):
  if i not in categorical_col:
    cont_var.append(i)

X = df[cont_var]
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))


# Converting categorical variables into One Hot Encoded values
categorical_col = ['workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'native-country']
for i in categorical_col:
  enc = OneHotEncoder(handle_unknown='ignore')
  enc.fit(df[i].values.reshape(-1,1))
  temp_df = pd.DataFrame(enc.transform(df[i].values.reshape(-1,1)).toarray())
  X = pd.concat([X, temp_df], axis=1)

# Load bin/cutpoints details
# Data available here - https://github.com/mlsquare/kitchen/blob/dert/scripts/dert/test_adult_bin_labels_1000.csv
bin_labels = pd.read_csv('../input/dert-adult-dataset/test_adult_bin_labels_1000.csv', delimiter=",",
                         header=0, names=['label', 'bins'])

Y = Y.reset_index().drop(columns='index')

# Load path details
# Available here - https://github.com/mlsquare/kitchen/blob/dert/scripts/dert/test_adult_paths_1000.csv
path_df = pd.read_csv('../input/dert-adult-dataset/test_adult_paths_1000.csv', delimiter=",",
                      header=0, names=['index', 'paths'])
path_df = path_df.drop(columns='index')

test_data = pd.concat([X.iloc[:1000,], Y.iloc[:1000,],path_df], axis=1)

# Append paths with 'S' and 'E'
new_path = []
for i, val in test_data.iterrows():
    new_path.append(val['paths'].split(sep=","))

_ = [x.insert(0, 'S') for x in new_path]
_ = [x.append('E') for x in new_path]

test_data['new_path'] = new_path
test_data = test_data.drop(["paths"], axis=1)

paths_lengths = np.array([len(xi) for xi in test_data.iloc[:,-1]])


## Create and train FFN

def _create_label_model(latent_dim=25, feature_size=104):
    input_layer = Input(shape=(feature_size,), name='ip_x')
    hidden_layer_x1 = Dense(10, activation='relu',
                            name='hidden_x1')(input_layer)
    hidden_layer_x2 = Dense(10, activation='relu',
                            name='hidden_x2')(hidden_layer_x1)
    hidden_layer_x3 = Dense(latent_dim, activation='relu',
                            name='hidden_x3')(hidden_layer_x2)
    output_layer = Dense(len(np.unique(Y)), activation='sigmoid',
                         name='op_x')(hidden_layer_x3)
    model = Model(input_layer, output_layer)
    return model


label_model = _create_label_model()

def fit_model():

    y_cat = to_categorical(Y)

    label_model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    label_model.fit(
        X, y_cat, batch_size=2000, epochs=50, verbose=1, shuffle=True, validation_split=0.2)

fit_model()

## Sequence data preperation for DeRT model

dir_indices = {
    'S': 0,
    'E': 1,
    '0': 2,
    '1': 3
}
bin_indices = {0:0, 1:1}
bin_indices.update({val: index+2 for index, val in enumerate(np.unique(bin_labels['label']))})

feature_indices = {'S':0, 'E': 15}
feature_indices.update({str(val): val for val in range(1,15)})

# Vectorization of paths 
feature_vocab_size = 16
bin_vocab_size = 882
dir_vocab_size = 4
latent_dim = 25

input_path_sequence = []
next_chars = []
features = []
paths_maxlen = np.max(paths_lengths)
# path_vocab_size = len(bin_labels) # How is this working? Validate!
# path_vocab_size = len(indices_label) # Temporary test for local trees
feature_size = 104
for i in range(0, len(test_data)):
# for i in range(0, len(shuffle_data[:140])):
    # get the feature
    curr_feat = np.array([test_data.iloc[i, 0:104]])
    curr_path = test_data.iloc[i, -1]
    curr_path_len = len(curr_path)
    for j in range(1, curr_path_len):
        features.append(curr_feat)
        input_path_sequence.append(curr_path[0:j])
        next_chars.append(curr_path[j])

# x_path = np.zeros((len(input_path_sequence), paths_maxlen, path_vocab_size), dtype=np.bool)

x_feat = np.zeros((len(input_path_sequence), paths_maxlen, feature_vocab_size), dtype=np.bool)

x_bin = np.zeros((len(input_path_sequence), paths_maxlen, bin_vocab_size), dtype=np.bool)

x_dir = np.zeros((len(input_path_sequence), paths_maxlen, dir_vocab_size), dtype=np.bool)



path_latent_input = np.zeros((len(input_path_sequence), feature_size), dtype=np.float)

# y_path = np.zeros((len(input_path_sequence), path_vocab_size), dtype=np.bool)

y_feat = np.zeros((len(input_path_sequence), feature_vocab_size), dtype=np.bool)

y_bin = np.zeros((len(input_path_sequence), bin_vocab_size), dtype=np.bool)

y_dir = np.zeros((len(input_path_sequence), dir_vocab_size), dtype=np.bool)

for i, sentence in enumerate(input_path_sequence):
    for t, char in enumerate(sentence):
        if char == 'S':
            x_feat[i, t, feature_indices[char]] = 1
            x_bin[i, t, 0] = 1
            x_dir[i, t, 0] = 1
        else:
            temp = re.compile("(\d+)(\w+)(\d+)") 
            res = temp.match(char).groups()
            x_feat[i, t, feature_indices[res[0]]] = 1
            x_bin[i, t, bin_indices[res[1]]] = 1
            x_dir[i, t, dir_indices[res[2]]] = 1
    if next_chars[i] == 'E':
        y_feat[i, feature_indices[next_chars[i]]] = 1
        y_bin[i, 1] = 1 ## Cross check
        y_dir[i, 1] = 1 ## Cross check
    else:
        temp = re.compile("(\d+)(\w+)(\d+)") 
        res = temp.match(next_chars[i]).groups()
        y_feat[i, feature_indices[res[0]]] = 1
        y_bin[i, bin_indices[res[1]]] = 1
        y_dir[i, dir_indices[res[2]]] = 1
    # y_path[i, label_indices[next_chars[i]]] = 1
    path_latent_input[i, :] = features[i]
    


## DeRT model architecture
from tensorflow.keras.layers import Reshape, Flatten, Bidirectional

label_model_latent = Input(shape=(latent_dim,), name='x_ip')

feature_input = Input(shape=(paths_maxlen, feature_vocab_size), name='feat_ip')

bin_input = Input(shape=(paths_maxlen, bin_vocab_size), name='bin_ip')

direction_input = Input(shape=(paths_maxlen, dir_vocab_size), name='dir_ip')

RNN = LSTM

merge_input = concatenate([feature_input, bin_input, direction_input], name='merge_ip')

decoder_1 = Bidirectional(RNN(latent_dim, return_state=False, return_sequences=True, name='lstm_1'))

decoder_1_outputs = decoder_1(merge_input, initial_state=[label_model_latent, label_model_latent, label_model_latent, label_model_latent])

decoder_2 = Bidirectional(RNN(latent_dim, return_state=False, name='lstm_2'))

decoder_2_outputs = decoder_2(decoder_1_outputs, initial_state=[label_model_latent, label_model_latent, label_model_latent, label_model_latent])

# hidden_1 = Dense(100, activation='softmax', name='h_1')(decoder_1_outputs)

# hidden_2 = Dense(100, activation='softmax', name='h_2')(hidden_1)

feat_hidden_1 = Dense(100, activation='softmax', name='f_1')(decoder_2_outputs)
output_feature = Dense(feature_vocab_size, activation='softmax', name='op_feat')(feat_hidden_1)

bin_hidden_1 = Dense(100, activation='softmax', name='b_1')(decoder_2_outputs)
bin_hidden_2 = Dense(100, activation='softmax', name='b_2')(bin_hidden_1)
output_bin = Dense(bin_vocab_size, activation='softmax', name='op_bin')(bin_hidden_2)

output_dir = Dense(dir_vocab_size, activation='softmax', name='op_dir')(decoder_2_outputs)

model = Model([label_model_latent, feature_input, bin_input, direction_input], [output_feature, output_bin, output_dir])


# Function to extract latent layer output from FFN
def get_hidden_x(x, model, layer_num=3):
    def get_hidden_x_inner(model, layer_num=layer_num):
        return K.function([model.layers[0].input], [model.layers[layer_num].output])
    return get_hidden_x_inner(model, layer_num=layer_num)([x])[0]

## Please use this code to load pretrained model
## from tensorflow.keras.models import load_model
# model = load_model('../bilstm_model_final.h5')

## Train DeRT model
x_latent = get_hidden_x(path_latent_input, model=label_model)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
				 loss_weights={'op_feat': 0.2,'op_bin': 1.0,'op_dir': 0.05})
model.fit([x_latent, x_feat, x_bin, x_dir], [y_feat, y_bin, y_dir],batch_size=128, epochs=9000, verbose=1)

