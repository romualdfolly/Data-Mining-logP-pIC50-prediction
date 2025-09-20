import numpy as np
from sklearn.svm import SVR
from models.model import Model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

class ProjectSVRegressor(Model):
    def __init__(self, X_train, X_test, X_val, y_train, y_test, y_val, kernels = ['rbf', 'linear'], reg_parameters = [1.0], gammas = [0.01, 0.1, 1], combine_train_and_val = True):

        # call to super class constructor
        super().__init__(X_train, X_test, X_val, y_train, y_test, y_val, combine_train_and_val)

        self.kernels = kernels
        self.gammas = gammas
        self.reg_parameters = reg_parameters

        self.best_params = None
    

    def find_optimal_reg_parameter(self):
        # Define the parameter grid for GridSearchCV
        param_grid = {
            'kernel': self.kernels,
            'C': self.reg_parameters,
            'gamma': self.gammas
        }

        # Create the SVR model
        svr = SVR()

        # Create the GridSearchCV object
        grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)

        # Fit the grid search to the combined training and validation data
        grid_search.fit(self.X_train, self.y_train)

        # Get the best estimator
        self.best_params = grid_search.best_params_

        # Print the best parameters
        print(f'\nBest parameters found: {self.best_params}')
    

    def train(self):
        if self.best_params is None:
            raise ValueError("Please find the optimal parameters first by calling find_optimal_parameter().")

        print(f"\n> Launch : SVR Regessor -> best parameters = {self.best_params}\n")
        self.model = SVR(**self.best_params)

        self.model.fit(self.X_train, self.y_train)

        ## Uncomment below and comment the one above to get the loss history

        # Capture the loss history
        #self.loss_history = []

        #n_samples = len(self.X_train)
        #for i in range(1, n_samples + 1):
        #    self.model.fit(self.X_train[:i], self.y_train[:i])
        #    y_pred = self.model.predict(self.X_train)
        #    loss_value = mean_squared_error(self.y_train, y_pred)
        #    self.loss_history.append(loss_value)

        self.plot_loss()