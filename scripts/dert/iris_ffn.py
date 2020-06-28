from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K


def _create_label_model(latent_dim=5):
    input_layer = Input(shape=(feature_size,), name='ip_x')
    hidden_layer_x1 = Dense(20, activation='tanh',
                            name='hidden_x1')(input_layer)
    hidden_layer_x2 = Dense(20, activation='tanh',
                            name='hidden_x2')(hidden_layer_x1)
    hidden_layer_x3 = Dense(latent_dim, activation='tanh',
                            name='hidden_x3')(hidden_layer_x2)
    output_layer = Dense(len(np.unique(y)), activation='softmax',
                         name='op_x')(hidden_layer_x3)
    model = Model(input_layer, output_layer)
    return model

## Create ffn
iris_ffn = _create_label_model()

# Train the ffn
y_cat = to_categorical(y)
iris_ffn.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
iris_ffn.fit(
    X, y_cat, batch_size=30, epochs=150, verbose=1, shuffle=True, validation_split=0.2)


# Function to extract hidden state of the ffn
def get_hidden_x(x, model, layer_num=3):
    def get_hidden_x_inner(model, layer_num=layer_num):
        return K.function([model.layers[0].input], [model.layers[layer_num].output])
    return get_hidden_x_inner(model, layer_num=layer_num)([x])[0]

# Hidden state used as initial state in DeRT-RNN model
x_ip = get_hidden_x(path_latent_input, model=label_model_trial_2)