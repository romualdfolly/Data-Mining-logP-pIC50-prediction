import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

class Model:
    """ MOTHER CLASS OF ALL OUR MODELS """
    def __init__(self, X_train, X_test, X_val, y_train, y_test, y_val, combine_train_and_val=True):

        self.combine_train_and_val = combine_train_and_val

        if self.combine_train_and_val:
            self.X_train = np.concatenate((X_train, X_val), axis=0)
            self.y_train = np.concatenate((y_train, y_val), axis=0)
        else:
            self.X_train = X_train
            self.y_train = y_train

        self.X_test = X_test
        self.X_val = X_val
        self.y_test = y_test
        self.y_val = y_val

        self.model = None

        self.loss_history = []


    def find_optimal_reg_parameter(self):
        ...

    def train(self):
        ...


    def predict(self, X):
        if X is None or X.size == 0:
            print(f"It has no missing value to predict here. shape of X = {X.shape}")
            return
        return self.model.predict(X)
    
    def plot_loss(self):
        if not self.loss_history:
            print("No loss history to plot. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Loss History')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss History Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

    

    def show_MSE_R2(self, X, y, type_ = 'Train'):
        # Make predictions
        y_pred = self.predict(X)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f'{type_} MSE: {mse}, {type_} R2: {r2}')


