from keras.layers import Dense, Input, concatenate, GRU, LSTM
from keras import backend as K
from keras.utils import to_categorical
from keras import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import jaccard_score
import numpy as np
import pandas as pd
import distance


class FeatureDrivenModel():

    def __init__(self):
        self.model = None

    def transform_data(self, X, y):
        # Create decision tree
        self.X = X
        self.y = y
        clf = DecisionTreeClassifier(random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.6, shuffle=True)
        clf.fit(X_train, y_train)
        left_nodes = clf.tree_.children_left[clf.tree_.children_left > 0]
        right_nodes = clf.tree_.children_right[clf.tree_.children_right > 0]
        cutpoints = np.around(clf.tree_.threshold, 1)[
            np.around(clf.tree_.threshold, 1) > -1]
        node_indicator = clf.decision_path(X)

        paths_as_int = []  # list of path for each instance in iris dataset. Change to paths_as_int
        for i, j in enumerate(X):
            paths_as_int.append(
                node_indicator.indices[node_indicator.indptr[i]:node_indicator.indptr[i+1]])

        # Convert path to strings
        paths_as_string = np.array([])
        decision_features_list = []
        cutpoints_list = []
        for i, j in enumerate(X):
            path = []
            decision_features = []
            cutpoints = []
            for node in paths_as_int[i]:
                if node == 0:
                    decision_features.append(0)
                    path.append('S')
                    decision_features.append(clf.tree_.feature[node]+1)
                    cutpoints.append(np.around(clf.tree_.threshold[node], 1))
                elif node in left_nodes:
                    path.append('L')
                    if clf.tree_.feature[node] >= 0:
                        decision_features.append(clf.tree_.feature[node]+1)
                        cutpoints.append(
                            np.around(clf.tree_.threshold[node], 1))
                elif node in right_nodes:
                    path.append('R')
                    if clf.tree_.feature[node] >= 0:
                        decision_features.append(clf.tree_.feature[node]+1)
                        cutpoints.append(
                            np.around(clf.tree_.threshold[node], 1))

            path.append('E')
            decision_features = np.array(decision_features)
            cutpoints = np.array(cutpoints)
            path = ' '.join(path)
            paths_as_string = np.append(paths_as_string, path)
            decision_features_list.append(decision_features)
            cutpoints_list.append(cutpoints)

        chars = ['S', 'L', 'R', 'E']

        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))

        X_new = np.hstack((X, paths_as_string.reshape(-1, 1)))
        # path_sequence = X_new[:, 4]
        data = pd.DataFrame(X_new)
        data[5] = y
        data[6] = np.array(decision_features_list)
        data[7] = np.array(cutpoints_list)
        df = data.sample(frac=1).reset_index(drop=True)
        self.df = df

        def get_path_lengths(t): return len(t.split())
        paths_lengths = np.array([get_path_lengths(xi)
                                  for xi in paths_as_string])

        # "+1" since 0 is a part of the vocab
        self.decision_feature_vocab_size = X.shape[1] + 1
        self.path_vocab_size = 4
        label_size = len(np.unique(y))
        self.feature_size = X.shape[1]

        self.paths_maxlen = np.max(paths_lengths)
        self.decision_features_maxlen = self.paths_maxlen-1

        input_path_sequence = []
        decision_feature_sequence = []
        extended_feature_sequence = []
        next_chars = []
        next_decision_feature = []
        features = []
        labels = []

        for i in range(0, len(df)):
            # get the feature
            curr_feat = np.array([df.iloc[i, 0:4]])
            curr_path = df.iloc[i, 4].split()
            curr_path_len = len(curr_path)
            curr_label = y[i]
            curr_dec_feat = df.iloc[i, 6]
            for j in range(1, curr_path_len):
                features.append(curr_feat)
                labels.append(curr_label)
                input_path_sequence.append(curr_path[0:j])
                next_chars.append(curr_path[j])
            for k in range(1, len(curr_dec_feat)):
                next_decision_feature.append(curr_dec_feat[k])
                decision_feature_sequence.append(curr_dec_feat[0:k])
            for k in range(0, len(curr_dec_feat)):
                extended_feature_sequence.append(curr_dec_feat[0:k+1])

        self.x_path = np.zeros(
            (len(input_path_sequence), self.paths_maxlen, self.path_vocab_size), dtype=np.bool)
        self.x_decision_features = np.zeros(
            (len(decision_feature_sequence), self.decision_features_maxlen, self.decision_feature_vocab_size), dtype=np.bool)

        self.path_latent_input = np.zeros(
            (len(input_path_sequence), self.feature_size), dtype=np.float)

        self.decision_feature_latent_input = np.zeros(
            (len(decision_feature_sequence), self.feature_size), dtype=np.float)

        self.decoder_input = np.zeros(
            (len(extended_feature_sequence), self.decision_features_maxlen, self.decision_feature_vocab_size), dtype=np.bool)

        self.y_path = np.zeros(
            (len(input_path_sequence), self.path_vocab_size), dtype=np.bool)
        self.y_decision_feature = np.zeros(
            (len(decision_feature_sequence), self.decision_feature_vocab_size), dtype=np.bool)

        for i, sentence in enumerate(input_path_sequence):
            for t, char in enumerate(sentence):
                self.x_path[i, t, self.char_indices[char]] = 1
            self.y_path[i, self.char_indices[next_chars[i]]] = 1
            self.path_latent_input[i, :] = features[i]

        for i, feat in enumerate(decision_feature_sequence):
            for t, val in enumerate(feat):
                self.x_decision_features[i, t, val] = 1
            self.y_decision_feature[i, next_decision_feature[i]] = 1
            self.decision_feature_latent_input[i, :] = features[i]

        for i, feat in enumerate(extended_feature_sequence):
            for t, val in enumerate(feat):
                self.decoder_input[i, t, val] = 1

    def create_model(self, **kwargs):
        self.path_model = self._create_path_model()
        self.feature_model = self._create_feature_model()
        self.label_model = self._create_label_model()

    def _create_label_model(self, latent_dim=5):
        input_layer = Input(shape=(self.feature_size,), name='ip_x')
        hidden_layer_x1 = Dense(20, activation='tanh',
                                name='hidden_x1')(input_layer)
        hidden_layer_x2 = Dense(20, activation='tanh',
                                name='hidden_x2')(hidden_layer_x1)
        hidden_layer_x3 = Dense(latent_dim, activation='tanh',
                                name='hidden_x3')(hidden_layer_x2)
        output_layer = Dense(3, activation='softmax',
                             name='op_x')(hidden_layer_x3)
        model = Model(input_layer, output_layer)
        return model

    def _create_feature_model(self, initialize=True, rnn_cell='gru', latent_dim=5):

        label_model_latent = Input(shape=(latent_dim,), name='label_ip')
        decision_feature_input = Input(shape=(
            self.decision_features_maxlen, self.decision_feature_vocab_size), name='dec_feat_ip')
        if rnn_cell == 'gru':
            RNN = GRU
        else:
            RNN = LSTM

        decoder = RNN(latent_dim, return_state=False,
                      return_sequences=False, name='gru_seq')
        if initialize:
            decoder_outputs = decoder(
                decision_feature_input, initial_state=label_model_latent)
        else:
            decoder_outputs = decoder(decision_feature_input)

        merge_layer = concatenate(
            [label_model_latent, decoder_outputs], name='cat')
        output_chars = Dense(self.decision_feature_vocab_size,
                             activation='softmax', name='op_sent')(merge_layer)
        model = Model(
            [label_model_latent, decision_feature_input], output_chars)
        return model

    def _create_path_model(self, initialize=True, rnn_cell='gru', latent_dim=5):

        # Hidden state from Ip to h3 layer of "label_model"
        label_model_latent = Input(shape=(latent_dim,), name='label_ip')
        # Decoder output from feature_model
        feature_model_decoder = Input(shape=(latent_dim,), name='feat_ip')
        path_input = Input(
            shape=(self.paths_maxlen, self.path_vocab_size), name='ip_sent')
        if rnn_cell == 'gru':
            RNN = GRU
        else:
            RNN = LSTM

        decoder = RNN(latent_dim, return_state=False,
                      return_sequences=False, name='gru_sent')
        if initialize:
            decoder_outputs = decoder(
                path_input, initial_state=label_model_latent)
        else:
            decoder_outputs = decoder(path_input)

        merge_layer = concatenate(
            [label_model_latent, feature_model_decoder, decoder_outputs], name='cat')
        output_chars = Dense(
            self.path_vocab_size, activation='softmax', name='op_sent')(merge_layer)
        model = Model([label_model_latent, feature_model_decoder,
                       path_input], output_chars)
        return model

    def get_hidden_x(self, x, model, layer_num=3):
        def get_hidden_x_inner(model, layer_num=layer_num):
            return K.function([model.layers[0].input], [model.layers[layer_num].output])
        return get_hidden_x_inner(model, layer_num=layer_num)([x])[0]

    def get_decoder_output(self, x1, x2, model, layer_num=2):
        temp_layer = K.function([model.layers[0].input, model.layers[1].input], [
                                model.layers[layer_num].output])
        return temp_layer([x1, x2])[0]

    def fit_model(self):

        y_cat = to_categorical(self.y)

        self.label_model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        label_model_history = self.label_model.fit(
            self.X, y_cat, batch_size=20, epochs=200, verbose=0, shuffle=True, validation_split=0.2)

        x_latent = self.get_hidden_x(
            self.decision_feature_latent_input, model=self.label_model)

        self.feature_model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.feature_model.fit([x_latent, self.x_decision_features], self.y_decision_feature,
                               batch_size=20, epochs=50, verbose=0, shuffle=True, validation_split=0.2)

        x_latent = self.get_hidden_x(self.path_latent_input, model=self.label_model)
        feat_dec_output = self.get_decoder_output(
            x_latent, self.decoder_input, self.feature_model)

        self.path_model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.path_model.fit([x_latent, feat_dec_output, self.x_path], self.y_path, batch_size=20,
                            epochs=200, verbose=0, shuffle=True, validation_split=0.2)

    def jaccard_score_inconsistent(self, x, y):
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)

    def get_j_coeff(self, a, b):
        if len(a) != len(b):
            return self.jaccard_score_inconsistent(a, b)
        return jaccard_score(a, b, average='micro')

    def predict(self, x):
        latent_dim = 5
        x_f = x.reshape(1, self.feature_size)
        token = 'S'
        feat_tok = 0  # Root node
        cont = True
        path = [token]
        feature_list = [feat_tok]
        x_path = np.zeros((1, self.paths_maxlen, self.path_vocab_size), dtype=np.bool)
        x_decision_features = np.zeros(
            (1, self.decision_features_maxlen, self.decision_feature_vocab_size), dtype=np.bool)
        decoder_input = np.zeros(
            (1, self.decision_features_maxlen, self.decision_feature_vocab_size), dtype=np.bool)

        x_latent = self.get_hidden_x(x_f, model=self.label_model)
        x_latent = x_latent.reshape(1, latent_dim)
        x_path[0, 0, self.char_indices[token]] = 1
        x_decision_features[0, 0, feat_tok] = 1
        decoder_input[0, 0, feat_tok] = 1
        pred = self.label_model.predict(x_f)
        label = [np.argmax(pred[0])]
        index = 1
        while (index < self.decision_features_maxlen):
            pred_feat = self.feature_model.predict([x_latent, x_decision_features])
            pred_val = np.argmax(pred_feat[0])
            x_decision_features[0, index, pred_val] = 1
            next_val = pred_val
            feature_list.append(next_val)
            decoder_input[0, index, feature_list[index]] = 1
            index += 1

        feat_decoder = self.get_decoder_output(
            x_latent, decoder_input, self.feature_model)
        index = 1
        while cont & (index < self.paths_maxlen):
            pred = self.path_model.predict([x_latent, feat_decoder, x_path])
            char_index = np.argmax(pred[0])
            x_path[0, index, char_index] = 1
            next_char = self.indices_char[char_index]
            path.append(next_char)
            index += 1
            if next_char == 'E':
                cont = False

        return [path, label, feature_list]

    def score(self):
        count = []
        j_coeff = []
        j_coeff_feat = []
        l_dist = []
        # pred_feat_list = []
        # pred_feat_accuracy = []
        bleu_score = []
        for i in range(self.df.shape[0]):
            curr_feat = np.array([self.df.iloc[i, 0:4]])
            path, label, decision_feature = self.predict(curr_feat)
            print('actual vs predicted: ', self.df.iloc[i, 4], ' vs ', ' '.join(
                path), 'labels: ', self.df.iloc[i, 5], label[0])
            count.append(self.df.iloc[i, 5] == label[0])
            actual_path = self.df.iloc[i, 4].split()
            actual_path_tok = [self.char_indices[char] for char in actual_path]
            pred_path_tok = [self.char_indices[char] for char in path]
            # print('actual_path--', actual_path)
            # print('path--', path)
            bleu_score.append(sentence_bleu([actual_path], path))
            j_coeff.append(self.get_j_coeff(actual_path_tok, pred_path_tok))
            j_coeff_feat.append(self.get_j_coeff(self.df.iloc[i, 6], decision_feature))
            l_dist.append(distance.levenshtein(
                self.df.iloc[i, 4].replace(' ', ''), ''.join(path)))

            print('Actual vs predicted features: ', self.df.iloc[i, 6], 'vs', decision_feature, '\n')


        print('\nLabel accuracy - ', np.mean(count))
        print('Path metric (Jaccard) - ', np.mean(j_coeff))
        print('Path metric (Levenshtein) - ', np.mean(l_dist))
        print('Decision feature metric (Jaccard) - ', np.mean(j_coeff_feat))
        print('Bleu score of paths - ', np.mean(bleu_score))



class CombinedModel(FeatureDrivenModel):

    def transform_data(self, X, y):
        super().transform_data(X, y)
        combined_list = []

        for i, j in enumerate(self.df.iloc[:,4]):
            char_list = []
            for index, char in enumerate(j.split()):
                if char == 'S':
                    char_list.append('S')
                elif char == 'E':
                    char_list.append('E')
                else:
                    # char_list.append((df.iloc[i,6][index], char)) # tuple option (4,R)
                    # char_list.append(df.iloc[i,6][index]) # int and char -- 4, R
                    # char_list.append(char)
                    char_list.append(str(self.df.iloc[i,6][index]) + char) # combined char '4R'
            combined_list.append(char_list)

        self.df[8] = combined_list

        chars = ['S', 'E']
        i=0
        while (i < self.X.shape[1]):
            chars.append(str(i+1)+'L')
            chars.append(str(i+1)+'R')
            i+=1

        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))

        self.path_vocab_size = len(chars)

        input_path_sequence = []
        next_chars = []
        features = []

        for i in range(0, len(self.df)):
            # get the feature
            curr_feat = np.array([self.df.iloc[i, 0:4]])
            curr_path = self.df.iloc[i, 8]
            curr_path_len = len(curr_path)
            # curr_label = y[i]
            # curr_dec_feat = df.iloc[i, 6]
            for j in range(1, curr_path_len):
                features.append(curr_feat)
                input_path_sequence.append(curr_path[0:j])
                next_chars.append(curr_path[j])

        self.x_path = np.zeros(
            (len(input_path_sequence), self.paths_maxlen, self.path_vocab_size), dtype=np.bool)

        self.path_latent_input = np.zeros(
            (len(input_path_sequence), self.feature_size), dtype=np.float)

        self.y_path = np.zeros(
            (len(input_path_sequence), self.path_vocab_size), dtype=np.bool)

        for i, sentence in enumerate(input_path_sequence):
            for t, char in enumerate(sentence):
                self.x_path[i, t, self.char_indices[char]] = 1
            self.y_path[i, self.char_indices[next_chars[i]]] = 1
            self.path_latent_input[i, :] = features[i]
        


    def create_model(self, **kwargs):
        self.combined_model = self._create_combined_model()
        self.label_model = super()._create_label_model()

    def _create_combined_model(self, initialize=True, rnn_cell='gru', latent_dim=5):

        label_model_latent = Input(shape=(latent_dim,), name='label_ip')
        path_input = Input(shape=(
            self.paths_maxlen, self.path_vocab_size), name='dec_feat_ip')
        if rnn_cell == 'gru':
            RNN = GRU
        else:
            RNN = LSTM

        decoder = RNN(latent_dim, return_state=False,
                      return_sequences=False, name='gru_seq')
        if initialize:
            decoder_outputs = decoder(
                path_input, initial_state=label_model_latent)
        else:
            decoder_outputs = decoder(path_input)

        merge_layer = concatenate(
            [label_model_latent, decoder_outputs], name='cat')
        output_chars = Dense(self.path_vocab_size,
                             activation='softmax', name='op_sent')(merge_layer)
        model = Model(
            [label_model_latent, path_input], output_chars)
        return model

    def fit_model(self):

        y_cat = to_categorical(self.y)

        self.label_model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.label_model.fit(
            self.X, y_cat, batch_size=20, epochs=200, verbose=0, shuffle=True, validation_split=0.2)

        x_latent = super().get_hidden_x(
            self.path_latent_input, model=self.label_model)

        self.combined_model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.combined_model.fit([x_latent, self.x_path], self.y_path,
                               batch_size=20, epochs=200, verbose=0, shuffle=True, validation_split=0.2)


    def predict(self, x):
        latent_dim = 5
        x_f = x.reshape(1, self.feature_size)
        token = 'S'
        cont = True
        path = [token]
        x_path = np.zeros((1, self.paths_maxlen, self.path_vocab_size), dtype=np.bool)

        x_latent = self.get_hidden_x(x_f, model=self.label_model)
        x_latent = x_latent.reshape(1, latent_dim)
        x_path[0, 0, self.char_indices[token]] = 1
        pred = self.label_model.predict(x_f)
        label = [np.argmax(pred[0])]
        index = 1
        print(self.paths_maxlen)
        while cont & (index < self.paths_maxlen):
            pred = self.combined_model.predict([x_latent, x_path])
            char_index = np.argmax(pred[0])
            x_path[0, index, char_index] = 1
            next_char = self.indices_char[char_index]
            path.append(next_char)
            index += 1
            if next_char == 'E':
                cont = False
            elif index == self.paths_maxlen - 1:
                path.append('E')

        return [path, label]


    def score(self):
        count = []
        bleu_score = []
        for i in range(self.df.shape[0]):
            curr_feat = np.array([self.df.iloc[i, 0:4]])
            path, label = self.predict(curr_feat)
            print('actual vs predicted: ', self.df.iloc[i, 8], ' vs ', ' '.join(
                path), 'labels: ', self.df.iloc[i, 5], label[0])
            count.append(self.df.iloc[i, 5] == label[0])
            path = list(''.join(path))
            actual_path = list(''.join(self.df.iloc[i, 8]))
            bleu_score.append(sentence_bleu([actual_path], path))

        print('Bleu score of paths - ', np.mean(bleu_score))
