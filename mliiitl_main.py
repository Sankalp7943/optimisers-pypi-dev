import numpy as np
import matplotlib as plt
import keras
import tensorflow as tf
import sys
import os
import shutil

class mliiitl:
    '''
    Creates mliiitl object from all the user data
    '''
    def __init__(self, x_train, y_train, x_test, y_test, 
                 model, loss, epoch, batch_size):
        try:
            self._x_train = x_train
            self._y_train = y_train
            self._x_test = x_test
            self._y_test = y_test
            self._model = model
            self._loss = loss
            self._epoch = epoch
            self._batch_size = batch_size
        except Exception:
            try:
                print('Invalid arguments given in mliiitl.__init__()')
            except Exception:
                pass
            try:
                print('Invalid arguments given in mliiitl.__init__()', file = sys.stdout)
            except Exception:
                pass 
        return self

    def delete_model_instance(temp):
        location = os.getcwd()
        folder = 'temp_model'
        path = os.path.join(location, folder)
        try:
            shutil.rmtree(path, ignore_errors = True)
        except Exception:
            print("Could not delete directory'temp_model',\
             Kindly delete the folder from current working directory.\
             May cause issues otherwise.")
    
    def save_output_model(arr_models, key):
        count = 1
        for model in arr_models:
            model.save('model_{model}'.format(model = key[count]))
            count += 1
        print('Models saved in {folder}'.format(folder = os.getcwd()))


    def save_model_instance(model):
        '''
        saves model
        '''
        model.save('temp_model')
        return 'temp_model'
    
    def splice_dataset_randomly(x_train,y_train):
        '''
        splices 1/8th data randomly for training
        '''
        number_of_rows = x_train.shape[0]
        random_indices = np.random.choice(number_of_rows, size=number_of_rows//8, replace=False)
        spliced_x_train = x_train[random_indices, :]
        spliced_y_train = y_train[random_indices, :]

        return spliced_x_train,spliced_y_train
    
    def test_performance(self, plots = False):
        '''

        '''
        temp = save_model_instance(self._model)
        spliced_x_train, spliced_y_train = splice_dataset_randomly(self._x_train, self._y_train)

        model_sgd = tf.keras.models.load_model('temp_model')
        model_rmsprop = tf.keras.models.load_model('temp_model')
        model_adagrad = tf.keras.models.load_model('temp_model')
        model_adadelta = tf.keras.models.load_model('temp_model')
        model_adam = tf.keras.models.load_model('temp_model')
        model_ftrl = tf.keras.models.load_model('temp_model')
        model_nadam = tf.keras.models.load_model('temp_model')
        model_adamax = tf.keras.models.load_model('temp_model')

        validation = (self._x_test, self._y_test)

        model_sgd.compile(optimizer = 'SGD', loss = self._loss, metrics = ['acc'])
        history_sgd = model_sgd.fit(spliced_x_train, spliced_y_train, epochs = self._epoch, batch_size = self._batch_size, validation_data = validation)

        model_rmsprop.compile(optimizer = 'RMSprop', loss = self._loss, metrics = ['acc'])
        history_rmsprop = model_rmsprop.fit(spliced_x_train, spliced_y_train, epochs = self._epoch, batch_size = self._batch_size, validation_data = validation)

        model_adagrad.compile(optimizer = 'Adagrad', loss = self._loss, metrics = ['acc'])
        history_adagrad = model_adagrad.fit(spliced_x_train, spliced_y_train, epochs = self._epoch, batch_size = self._batch_size, validation_data = validation)

        model_adadelta.compile(optimizer = 'Adadelta', loss = self._loss, metrics = ['acc'])
        history_adadelta = model_adagrad.fit(spliced_x_train, spliced_y_train, epochs = self._epoch, batch_size = self._batch_size, validation_data = validation)

        model_adam.compile(optimizer = 'adam', loss = self._loss, metrics = ['acc'])
        history_adam = model_adam.fit(spliced_x_train, spliced_y_train, epochs = self._epoch, batch_size = self._batch_size, validation_data = validation)

        model_ftrl.compile(optimizer = 'Ftrl', loss = self._loss, metrics = ['acc'])
        history_ftrl = model_ftrl.fit(spliced_x_train, spliced_y_train, epochs = self._epoch, batch_size = self._batch_size, validation_data = validation)

        model_nadam.compile(optimizer = 'Nadam', loss = self._loss, metrics = ['acc'])
        history_nadam = model_nadam.fit(spliced_x_train, spliced_y_train, epochs = self._epoch, batch_size = self._batch_size, validation_data = validation)

        model_adamax.compile(optimizer = 'Adamax', loss = self._loss, metrics = ['acc'])
        history_adamax = model_adamax.fit(spliced_x_train, spliced_y_train, epochs = self._epoch, batch_size = self._batch_size, validation_data = validation)
        
        delete_model_instance(temp)
        output = [history_sgd, history_rmsprop, history_adagrad, history_adadelta, history_adam, history_ftrl, history_nadam, history_adamax]

        print("1:'SGD', 2:'RMSprop', 3:'AdaGrad', 4:'AdaDelta', 5:'Adam', 6:'Ftrl', 7:'Nadam', 8:'Adamax'")
        key = {1:'SGD', 2:'RMSprop', 3:'AdaGrad', 4:'AdaDelta', 5:'Adam', 6:'Ftrl', 7:'Nadam', 8:'Adamax'}
        
        if save:
            arr_models = [model_sgd, model_rmsprop, model_adagrad, model_adadelta, model_adam,
             model_ftrl, model_nadam, model_adamax]
            save_output_model(arr_models, key)

        if plots:
            get_plots(output)
            return output
        else:
            return output

    def get_plots(output):
        key = {1:'SGD', 2:'RMSprop', 3:'AdaGrad', 4:'AdaDelta', 5:'Adam', 6:'Ftrl', 7:'Nadam', 8:'Adamax'}
        count = 1
        for history in output:
            plt.plot(history.history['acc'], label = key[count])
            count += 1
        plt.title('Model Training Accuracy')
        plt.ylabel('Training Accuracy')
        plt.xlabel('Epoch(s)')
        plt.legend()
        plt.figure(figsize = (15,10))
        plt.show()
        
        count = 1
        for history in output:
            plt.plot(history.history['val_acc'], label = key[count])
            count += 1
        plt.title('Model Validation Accuracy')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Epoch(s)')
        plt.legend()
        plt.figure(figsize = (15,10))
        plt.show()

        count = 1
        for history in output:
            plt.plot(history.history['loss'], label = key[count])
            count += 1
        plt.title('Model Training Loss')
        plt.ylabel('Training Loss')
        plt.xlabel('Epoch(s)')
        plt.legend()
        plt.figure(figsize = (15,10))
        plt.show()

        count = 1
        for history in output:
            plt.plot(history.history['val_loss'], label = key[count])
            count += 1
        plt.title('Model Validation Loss')
        plt.ylabel('Validation Loss')
        plt.xlabel('Epoch(s)')
        plt.legend()
        plt.figure(figsize = (15,10))
        plt.show()
       

