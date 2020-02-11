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
        self.clf = clf #Temporary, remove after use!!!
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
        self.data = data
        data[data.shape[1]] = y
        data[data.shape[1]] = np.array(decision_features_list)
        data[data.shape[1]] = np.array(cutpoints_list)
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

        # self.decision_feature_latent_input = np.zeros(
        #     (len(decision_feature_sequence), self.feature_size), dtype=np.float)

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

        # for i, feat in enumerate(decision_feature_sequence):
        #     for t, val in enumerate(feat):
        #         self.x_decision_features[i, t, val] = 1
        #     self.y_decision_feature[i, next_decision_feature[i]] = 1
            # print(features)
            # self.decision_feature_latent_input[i, :] = features[i]

        # for i, feat in enumerate(extended_feature_sequence):
        #     for t, val in enumerate(feat):
        #         self.decoder_input[i, t, val] = 1

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
        output_layer = Dense(len(np.unique(self.y)), activation='softmax',
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

        for i, j in enumerate(self.df.iloc[:,X.shape[1]]):
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
                    char_list.append(str(self.df.iloc[i,X.shape[1]+2][index]) + char) # combined char '4R'
            combined_list.append(char_list)

        # print("Combined list -- ",combined_list)
        
        self.df[self.df.shape[1]] = combined_list

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
            curr_feat = np.array([self.df.iloc[i, 0:X.shape[1]]])
            curr_path = self.df.iloc[i, -1]
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

        print(input_path_sequence)
        print(len(input_path_sequence))
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
            self.X, y_cat, batch_size=200, epochs=2, verbose=1, shuffle=True, validation_split=0.2)

        x_latent = super().get_hidden_x(
            self.path_latent_input, model=self.label_model)

        self.combined_model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.combined_model.fit([x_latent, self.x_path], self.y_path,
                               batch_size=200, epochs=2, verbose=1, shuffle=True)


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
        while cont & (index < self.paths_maxlen):
            pred = self.combined_model.predict([x_latent, x_path])
            char_index = np.argmax(pred[0])
            x_path[0, index, char_index] = 1
            next_char = self.indices_char[char_index]
            path.append(next_char)
            index += 1
            if next_char == 'E':
                cont = False
            # elif index == self.paths_maxlen - 1:
            #     path.append('E')

        if path[-1] != 'E':
            path.append('E')

        return [path, label]

    def check_path(self, path): # Returns -1 if path traversed path is wrong/non-existant
        # path = ''.join(path)
        path = path[1:-1]
        pred_features = []
        path_as_strings = []
        for i in range(len(path)):
            pred_features.append(int(path[i][:-1]))
            path_as_strings.append(path[i][-1])
            # if i%2 == 0:
            #     print('i -- ', i)
            #     print('path -- ', path)
            #     print('path[i] -- ', path[i])
            #     pred_features.append(int(path[i]))
            # else:
            #     path_as_strings.append(path[i])

        n_nodes = self.clf.tree_.node_count
        children_left = self.clf.tree_.children_left
        children_right = self.clf.tree_.children_right
        feature = self.clf.tree_.feature

        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            # node_depth[node_id] = parent_depth + 1

            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
                

        node = 0
        pred_target = -1
        subset_path = False
        for i in range(len(path_as_strings)):
            if path_as_strings[i] == 'L':
                if feature[node]+1 == pred_features[i]:
                    node = children_left[node]
                # else:
                    # pred_target = -1 # Remove for "subset" checks
                    # break
            elif path_as_strings[i] == 'R':
                if feature[node]+1 == pred_features[i]:
                    node = children_right[node]
                # else:
                    # pred_target = -1 # Remove for "subset" checks
                    # break
            if is_leaves[node]:
                for i, x in enumerate(self.clf.tree_.value[node][0]):
                    if x > 0:
                        pred_target = i
                if i < len(path_as_strings):
                    subset_path = True

        return pred_target, subset_path


    def score(self):
        count = []
        bleu_score = []
        j_coeff = []
        l_dist = []
        path_mismatch_count = []
        traverse_check_count = []
        order_mismatch_count = []
        subset_path_count = []
        for i in range(self.df.shape[0]):
            curr_feat = np.array([self.df.iloc[i, 0:self.X.shape[1]]])
            path, label = self.predict(curr_feat)
            actual_path = self.df.iloc[i, -1]

            actual_path_tok = [self.char_indices[char] for char in actual_path]
            pred_path_tok = [self.char_indices[char] for char in path]

            j_coeff.append(super().get_j_coeff(actual_path_tok, pred_path_tok))

            print('actual vs predicted: ', self.df.iloc[i, -1], ' vs ', ' '.join(
                path), 'labels: ', self.df.iloc[i, self.X.shape[1]+1], label[0])
            count.append(self.df.iloc[i, self.X.shape[1]+1] == label[0])
            # print('Actual path -- ', actual_path)
            # print('Pred path -- ', path)
            if actual_path != path:
                print(' -- Path mismatch -- ')
                if sorted(actual_path) == sorted(path):
                    print(' -- Order mismatch -- ')
                    order_mismatch_count.append(1)
                else:
                    path_mismatch_count.append(1)
                    pred_target, subset_path = self.check_path(path)
                    subset_path_count.append(subset_path)
                    if pred_target != -1 and pred_target == self.df.iloc[i, self.X.shape[1]+1]:
                        traverse_check_count.append(1)


            path = list(''.join(path))
            actual_path = list(''.join(self.df.iloc[i, -1]))
            bleu_score.append(sentence_bleu([actual_path], path))

            lev_path = []
            for i in range(len(path)):
                if i in ['S','L','R','E']:
                    lev_path.append(i)
            l_dist.append(distance.levenshtein(
                self.df.iloc[i, self.X.shape[1]].replace(' ', ''), ''.join(lev_path)))


        print('\nLabel accuracy - ', np.mean(count))
        print('Path metric (Jaccard) - ', np.mean(j_coeff))
        print('Path metric (Levenshtein) - ', np.mean(l_dist))
        print('Path mismatch count - ', np.sum(path_mismatch_count))
        print('Right traverse count - ', np.sum(traverse_check_count))
        print('Order mismatch count - ', np.sum(order_mismatch_count))
        print('Subset path count - ', np.sum(subset_path_count))
        print('Bleu score of paths - ', np.mean(bleu_score))
