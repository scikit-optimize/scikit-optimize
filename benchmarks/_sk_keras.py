import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import r2_score
from keras.regularizers import l2


class _KerasModel():
    def __init__(self,
                 classification=False,
                 n_neurons=128,
                 n_layers=2,
                 neuron_type='relu',
                 opt_par_1=0.001,
                 opt_par_2=0.9,
                 opt_par_3=0.99,
                 optim_type='adam',
                 batch_size=128,
                 dropout=0.0,
                 l2_regularization=0.00001):
        """
        This implements deep fully connected neural network class with sklearn-like interface

        :param classification: Type of task - classification or regression
        :param n_neurons: Number of neurons in every layer
        :param n_layers: Number of layers in neural network
        :param neuron_type: Type of neuron, could be 'relu' or 'tanh'
        :param optim_type: Type of optimizer to use, 'rmsprop' or 'adam'
        :param opt_par_1: Learning rate
        :param opt_par_2: Rho for rmsprop, Beta1 for adam
        :param opt_par_3: Beta2 for adam
        :param batch_size: Size of data subset that is used to update the model
        :param dropout: dropout regularization value
        :param l2_regularization: regularization imposed on weights of neural network

        """
        self.config = n_neurons, n_layers, neuron_type, opt_par_1, opt_par_2, opt_par_3, optim_type, batch_size, dropout, l2_regularization
        self.classification = classification # remember if the model should do classification or regression

    def fit(self, X, Y):
        """
        Fits regressor nn to input data.

        :param X: A matrix with every row being example input
        :param Y: A vector with every entry being example output

        :return: None
        """

        # convert both X and Y to float32
        X = X.astype('float32')

        if self.classification:
            Y = Y.astype('int32')
        else:
            Y = Y.astype('float32')

        n_neurons, n_layers, neuron_type, opt_par_1, opt_par_2, opt_par_3, optim_type, batch_size, dropout, l2_regularization = self.config
        n_features = X.shape[1]

        ip = Input(shape=(n_features,), dtype='float32')

        x = ip

        for i in range(n_layers):
            x = Dense(n_neurons, activation=neuron_type, W_regularizer=l2(l2_regularization))(x)
            #x = Dropout(p=dropout)(x)

        if self.classification:
            y = Dense(len(np.unique(Y)), activation='softmax', W_regularizer=l2(l2_regularization))(x)
        else:
            y = Dense(1, W_regularizer=l2(l2_regularization))(x)

        self.model = Model(input=ip, output=y)

        opt = None # variable for optimizer
        if optim_type == "adam":
            opt = Adam(lr=opt_par_1, beta_1=opt_par_2, beta_2=opt_par_3)
        elif optim_type == 'rmsprop':
            opt = RMSprop(lr=opt_par_1, rho=opt_par_2)

        if self.classification:
            self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
        else:
            self.model.compile(optimizer=opt, loss='mse')

        # split data into training and validation to use for stopping criterion
        Tr = int(len(X) * 0.7) # use 70% for training, rest for valiation

        X, Xv = X[:Tr], X[Tr:]
        Y, Yv = Y[:Tr], Y[Tr:]

        max_patience = 16
        patience = max_patience
        best_perf = -np.inf
        best_weights = None

        for epoch in range(64): # limit the number of epochs
            self.model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0)
            perf = self.score(Xv, Yv)

            if best_perf < perf:
                best_perf = perf
                best_weights = self.model.get_weights()
                patience = max_patience
            else:
                patience -= 1

            if patience <= 0:
                break

        self.model.set_weights(best_weights)

    def score(self, X, Y):
        Yp = self.model.predict(X)

        if self.classification:
            idx = np.argmax(Yp, axis=1)
            perf = np.mean(Y == idx) # accuracy
        else:
            perf = r2_score(Y, Yp) # r squared

        return perf


class KerasRegressor(_KerasModel):
    def __init__(self,
                 n_neurons=128,
                 n_layers=2,
                 neuron_type='relu',
                 opt_par_1=0.001,
                 opt_par_2=0.9,
                 opt_par_3=0.99,
                 optim_type='adam',
                 batch_size=128,
                 dropout=0.0,
                 l2_regularization=0.00001):
        _KerasModel.__init__(self, False, n_neurons,
                             n_layers,
                             neuron_type,
                             opt_par_1,
                             opt_par_2,
                             opt_par_3,
                             optim_type,
                             batch_size,
                             dropout,
                             l2_regularization)


class KerasClassifier(_KerasModel):
    def __init__(self,
                 n_neurons=128,
                 n_layers=2,
                 neuron_type='relu',
                 opt_par_1=0.001,
                 opt_par_2=0.9,
                 opt_par_3=0.99,
                 optim_type='adam',
                 batch_size=128,
                 dropout=0.0,
                 l2_regularization=0.00001):
        _KerasModel.__init__(self, True, n_neurons,
                             n_layers,
                             neuron_type,
                             opt_par_1,
                             opt_par_2,
                             opt_par_3,
                             optim_type,
                             batch_size,
                             dropout,
                             l2_regularization)
