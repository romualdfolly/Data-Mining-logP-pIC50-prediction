from models.model import Model
import pandas as pd, numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class ProjectLinearRegressor(Model):
    def __init__(
            self, X_train, X_test, X_val, y_train, y_test, y_val,
            reg_paremeters = [0.001, 0.01, 0.1, 1, 10, 100], reg_mod = "lasso", max_iter=10000, combine_train_and_val = True
        ):

        # call to super class constructor
        super().__init__(X_train, X_test, X_val, y_train, y_test, y_val, combine_train_and_val)


        self.reg_mod = reg_mod
        self.reg_parameters = reg_paremeters
        self.optimal_reg_parameter = None
        self.max_iter = max_iter
    

    def find_optimal_reg_parameter(self):
        # Paramètres pour la recherche de grille
        reg_params = {
            'alpha': self.reg_parameters
        }

        # regularization Model instanciation
        reg_model = None
        if self.reg_mod.lower() == 'lasso':
            reg_model = Lasso(max_iter= self.max_iter)
        elif self.reg_mod.lower() == 'ridge':
            reg_model = Ridge(max_iter= self.max_iter)
        
        # Regularization model control
        if reg_model == None:
            raise ValueError("Warning ! You need to choose Lasso or Ridge as regularization mode")

        grid_search = GridSearchCV(reg_model, reg_params, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_train, self.y_train)

        # mean scores print
        df = pd.DataFrame({'param': [param['alpha'] for param in grid_search.cv_results_['params']], 'mean_test_score': -grid_search.cv_results_['mean_test_score']}).sort_values(by='mean_test_score', ascending=False)
        print(df)

        print(f"\nThe best regularization parameter is {grid_search.best_estimator_}")

        # Best model parameter
        self.optimal_reg_parameter = grid_search.best_estimator_
    


    def train(self):
        if self.optimal_reg_parameter is None:
            raise ValueError("Please find the optimal parameters first by calling find_optimal_parameter().")

        print("\n> Launch : LinearRegression\n")
        # Entraîner le modèle de régression linéaire
        self.model = LinearRegression()
        
        self.model.fit(self.X_train, self.y_train)

        ## Uncomment below and comment the one above to get the loss history
        
        #self.loss_history = []

        #n_samples = len(self.X_train)
        #for i in range(1, n_samples + 1):
        #    self.model.fit(self.X_train[:i], self.y_train[:i])
        #    y_pred = self.model.predict(self.X_train)
        #    loss_value = mean_squared_error(self.y_train, y_pred)
        #    self.loss_history.append(loss_value)
        
        
        self.plot_loss()
