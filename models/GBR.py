import numpy as np
from models.model import Model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score



class ProjectGBRegressor(Model):
    def __init__(
            self, X_train, X_test, X_val, y_train, y_test, y_val,
            learning_rates = [0.01, 0.1, 1], max_depths = list(range(10, 20, 5)), n_estimators = list(range(10, 30, 10)), sub_samples = np.arange(0.5, 1.0, 0.1),
            combine_train_and_val = True
        ):

        # call to super class constructor
        super().__init__(X_train, X_test, X_val, y_train, y_test, y_val, combine_train_and_val)


        self.n_estimators = n_estimators
        self.max_depths = max_depths
        self.learning_rates = learning_rates
        self.sub_samples = sub_samples

        self.best_params = None
    
    
    def find_optimal_parameter(self):

        # Randomized parameters definition
        param_grid = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depths,
            'learning_rate': self.learning_rates,
            'subsample': self.sub_samples,
        }


        # Decision Tree Estimator's instanciation
        tree = GradientBoostingRegressor(random_state=42)

        # GridSearchCV object Instanciation
        grid_tree = RandomizedSearchCV(tree, param_grid, cv=5, n_iter=5, scoring='neg_mean_squared_error')

        # GridSearchCV object Training
        grid_tree.fit(self.X_train, self.y_train)

        # set of best parameters
        self.best_params = grid_tree.best_params_

        # Optimal paremeters display
        print("Optimal parameters:", self.best_params)


    def train(self):
        if self.best_params is None:
            raise ValueError("Please find the optimal parameters first by calling find_optimal_parameter().")

        print(f"\n> Launch : GB Regessor -> best parameters = {self.best_params}\n")

        self.model = GradientBoostingRegressor(**self.best_params)

        self.model.fit(self.X_train, self.y_train)

        # Capture the loss history
        self.loss_history = []
        for y_pred in self.model.staged_predict(self.X_train):
            loss_value = mean_squared_error(self.y_train, y_pred)
            self.loss_history.append(loss_value)

        self.plot_loss()
