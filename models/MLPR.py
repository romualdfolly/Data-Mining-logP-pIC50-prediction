import sys, io
import numpy as np
from models.model import Model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ProjectMLPRegressor(Model):
    def __init__(self, X_train, X_test, X_val, y_train, y_test, y_val, activation = 'relu', hidden_layers = (100, ), combine_train_and_val = True):

        
        # call to super class constructor
        super().__init__(X_train, X_test, X_val, y_train, y_test, y_val, combine_train_and_val)


        self.activation = activation
        self.hidden_layers = hidden_layers
    


    def train(self):
        print(f"\n> Launch : MLP Regessor -> hidden-layer = {self.hidden_layers}\n")

        # Capture the console output for loss history
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout


        # Create the MLPRegressor model
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layers, activation=self.activation, solver='adam', max_iter=1000, verbose=True, random_state=42)

        # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Restore the console output
        sys.stdout = old_stdout
        loss_output = new_stdout.getvalue()
        

        # Parse the loss history
        for line in loss_output.split('\n'):
            if 'loss' in line:
                try:
                    loss_value = float(line.split()[-1])
                    self.loss_history.append(loss_value)
                except ValueError:
                    continue
        
        self.plot_loss()
