import random
import time

import keras.models
import matplotlib.pyplot as plt
import pandas as pd
# ----intialization of random seeds
# os.environ['PYTHONHASHSEED'] = '0'
# random.seed(12345)
# np.random.seed(42)
# -----------import tensor libraries---
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten, MaxPool1D, BatchNormalization, Activation, ReLU
from tensorflow.keras.models import Sequential

tf.random.set_seed(1234)
from tensorflow.keras.optimizers import  Adam
from performance_eval import error_stats

class model_cnn():
    def __init__(self):
        self.time_to_save=time.strftime("%d-%m-%Y-%H-%M-%S")
        self.model_acc = []

    # generate model
    def build_cnn_model(self, input_shape, learn_rate=0.001, pool_size=2, layers=[64, 128], activation='relu'):
        # create model
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=5, activation=activation, input_shape=input_shape))
        model.add(Conv1D(filters=64, activation=activation, kernel_size=3))
        model.add(MaxPool1D(pool_size=pool_size))
        model.add(Flatten())
        for idx, output in enumerate(layers):
            if activation == 'tanh':
                model.add(Dense(output, activation=activation, use_bias=False))
                model.add(BatchNormalization())
            else:
                model.add(Dense(output, use_bias=False))
                model.add(BatchNormalization())
                model.add(Activation(activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    # train model that is built from randomly selected configuration
    def get_best_model(self, val_x, val_y, x_test, y_test, cnn_params, n_search=20):
        # inputs for validation and test are converted to dataframes
        self.val_x = pd.DataFrame(val_x)
        self.val_y = pd.DataFrame(val_y)
        x_test = pd.DataFrame(x_test)
        y_test = pd.DataFrame(y_test)
        # reshape val_x and x_test
        self.val_x_shaped = self.val_x.values.reshape((self.val_x.shape[0], self.val_x.shape[1], 1))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))
        param_dicts = []
        for i in range(n_search):
            temp_dict = {}
            for key in cnn_params:
                rand_val = random.sample(cnn_params[key], 1)  # returns list
                temp_dict[key] = rand_val[0]
            param_dicts.append(temp_dict)
        # print(param_dicts)
        acc_lst = []
        for idx, dict in enumerate(param_dicts):
            model = self.build_cnn_model(input_shape=(self.val_x.shape[1], 1),
                                         learn_rate=dict["learn_rate"],
                                         layers=dict["layers"])
            # print(dict)
            self.history = model.fit(x=self.val_x_shaped, y=self.val_y, batch_size=dict['batch_size'],
                      epochs=dict['epochs'], shuffle=True, validation_split=.2, verbose=0)
            _, accuracy = model.evaluate(x_test, y_test, verbose=0)
            acc_lst.append(accuracy)
        self.best_params = param_dicts[acc_lst.index(max(acc_lst))]
        # build the best model
        self.model = self.build_cnn_model(input_shape=(self.val_x.shape[1], 1),
                                          learn_rate=self.best_params["learn_rate"],
                                          layers=self.best_params["layers"])

    def cnn_train(self, trainX, trainy_label, epoch=35, batch_size=512, verbose=0):
        # inputs for training cnn is dataset and lables
        # ***********First Run model_cnn().get_best_model to get the best model**********
        self.trainX = pd.DataFrame(trainX)
        self.trainy = pd.DataFrame(trainy_label)

        self.X_train_shaped = self.trainX.values.reshape(
            (self.trainX.shape[0], self.trainX.shape[1], 1))  # important: inputs should be reshaped for CNN

        self.history = self.model.fit(self.X_train_shaped, self.trainy, shuffle=True, validation_split=.2,
                                      batch_size=batch_size, epochs=epoch, verbose=verbose)

    def cnn_train_static(self, trainX, trainy_label, lr=0.001, epoch=40, pool_size=2, batch_size=512, verbose=0):
        # inputs for training cnn is dataset and lables
        self.trainX = pd.DataFrame(trainX)
        self.trainy = pd.DataFrame(trainy_label)
        self.learning_rate = lr
        self.X_train_shaped = self.trainX.values.reshape((self.trainX.shape[0], self.trainX.shape[1], 1))
        self.model = Sequential()
        self.model.add(
            Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(self.X_train_shaped.shape[1], 1)) )
        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(MaxPool1D(pool_size=pool_size))
        self.model.add(Flatten())
        self.model.add(Dense(100, use_bias=False))
        ReLU()
        BatchNormalization()
        self.model.add(Dropout(0.2))
        self.model.add(Dense(500, use_bias=False))
        ReLU()
        BatchNormalization()
        self.model.add(Dropout(0.2))
        self.model.add(Dense(500, use_bias=False))
        ReLU()
        BatchNormalization()
        self.model.add(Dropout(0.2))
        self.model.add(Dense(100, use_bias=False))
        ReLU()
        BatchNormalization()
        self.model.add(Dropout(0.2))
        self.model.add(Dense(2, activation='linear'))  # The output layer
        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate),
                           metrics=["mse"])
        # print('Model Summary', self.model.summary())
        # print("Learning Rate", self.learning_rate)
        self.history = self.model.fit(self.X_train_shaped, self.trainy, shuffle=True, validation_split=.2,
                                      batch_size=batch_size, epochs=epoch, verbose=verbose)

    def cnn_train_on_batch(self, trainX, trainy_label):
        # inputs for training cnn is dataset and lables
        self.trainX = pd.DataFrame(trainX)
        self.trainy = pd.DataFrame(trainy_label)

        self.X_train_shaped = self.trainX.values.reshape(
            (self.trainX.shape[0], self.trainX.shape[1], 1))  # important: inputs should be reshaped for CNN
        self.model.train_on_batch(self.X_train_shaped, self.trainy)

    def cnn_test(self,testX,batch_size=512):
        testX_shaped = testX.values.reshape((testX.shape[0], testX.shape[1], 1))
        # yhat= self.model.predict_classes(testX_shaped,batch_size=batch_size) #we can also use self.model.predict which returns probabilistic output between 0 and 1
        yhat = self.model.predict(testX_shaped, batch_size=batch_size)
        yhat = (yhat > 0.5).astype("int32")
        return yhat

    def cnn_proba(self,testX,batch_size=512):
        testX_shaped = testX.values.reshape((testX.shape[0], testX.shape[1], 1))
        proba = self.model.predict(testX_shaped, batch_size=batch_size)
        return proba

    def cnn_acc(self):
        _, accuracy = self.model.evaluate(self.yhat, self.testy, batch_size=512, verbose=0)
        return accuracy


    def plot_loss_validation(self,addr_2save=None):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if addr_2save!= None :
            plt.savefig(addr_2save)
        else:
            plt.show()
        plt.close()

    def plot_accuracy(self,addr_2save=None):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if addr_2save != None:
            plt.savefig(addr_2save)
        else:
            plt.show()
        plt.close()


    def plot_roc(self,addr_2save=None):

        fpr, tpr, thresholds = metrics.roc_curve(self.testy, self.yhat, pos_label=1)
        # plt.plot(fpr,tpr)
        # plt.show()
        plt.plot(fpr, tpr, marker='.', label='CNN')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        if addr_2save != None:
            plt.savefig(addr_2save)
        else:
            plt.show()
        plt.close()

    def save_cnn_model(self, address, name):
        self.model.save(address+name)

    def load_cnn_model(self, address, name):
        self.model = keras.models.load_model(address+name)

    def save_prediction(self,address):
        #save 3 things, prediction results, prediction scores, true label
        pass

    def save_model_report(self,address):
        print('CNN Architecture Summary', self.model_summary,file=open(address +'cnn_report_'+ self.time_to_save + ".txt", "a"))


def main():
    # Load data
    x_w_train = pd.read_csv('files/x_w_train.csv', index_col=0)
    x_train = x_w_train.drop(columns=['timestamp', 'gw_ref'])
    x_w_test = pd.read_csv('files/x_w_test.csv', index_col=0)
    x_test = x_w_test.drop(columns=['timestamp', 'gw_ref'])
    y_w_train = pd.read_csv('files/y_w_train.csv', index_col=0)
    y_w_test = pd.read_csv('files/y_w_test.csv', index_col=0)

    # Build model
    cnn_model = model_cnn()
    print('training model.............')
    cnn_model.cnn_train_static(x_train, y_w_train)
    print('testing.............')
    y_pred = cnn_model.cnn_test(x_test)
    print('Calculating error...............')
    print(error_stats(y_pred, y_w_test))

if __name__ == '__main__':
    main()