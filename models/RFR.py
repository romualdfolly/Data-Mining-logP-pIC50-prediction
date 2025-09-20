import numpy as np
from random import randint
from models.model import Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score



class ProjectRFRegressor(Model):
    def __init__(
            self, X_train, X_test, X_val, y_train, y_test, y_val,
            n_estimators = list(range(10, 100)), max_depths = list(range(10, 50)), min_samples_splits = list(range(2, 20)), min_samples_leafs = list(range(1, 20)),
            combine_train_and_val = True
        ):

        # call to super class constructor
        super().__init__(X_train, X_test, X_val, y_train, y_test, y_val, combine_train_and_val)


        self.X_test = X_test
        self.X_val = X_val
        self.y_test = y_test
        self.y_val = y_val

        self.n_estimators = n_estimators
        self.max_depths = max_depths
        self.min_samples_leafs = min_samples_leafs
        self.min_samples_splits = min_samples_splits

        self.best_params = None

    

    def find_optimal_parameter(self):

        # Randomized parameters definition
        param_grid = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depths,
            'min_samples_split': self.min_samples_splits,
            'min_samples_leaf': self.min_samples_leafs,
        }


        # Decision Tree Estimator's instanciation
        tree = RandomForestRegressor(random_state=42)

        # GridSearchCV object Instanciation
        grid_tree = RandomizedSearchCV(tree, param_grid, cv=5, scoring='neg_mean_squared_error')

        # GridSearchCV object Training
        grid_tree.fit(self.X_train, self.y_train)

        # set of best parameters
        self.best_params = grid_tree.best_params_

        # Optimal parameters display
        print("Optimal parameters :", grid_tree.best_params_)


    def train(self):
        if self.best_params is None:
            raise ValueError("Please find the optimal parameters first by calling find_optimal_parameter().")

        print(f"\n> Launch : RF Regessor -> best parameters = {self.best_params}\n")
        self.model = RandomForestRegressor(**self.best_params)

        self.model.fit(self.X_train, self.y_train)


        # Capture the loss history
        self.loss_history = []
        for i, tree in enumerate(self.model.estimators_):
            partial_model_params = {k: v for k, v in self.best_params.items() if k != 'n_estimators'}  # +++
            partial_model = RandomForestRegressor(n_estimators=i + 1, **partial_model_params, oob_score=True, random_state=42)  # +++
            partial_model.estimators_ = self.model.estimators_[:i + 1]
            y_pred = np.mean([est.predict(self.X_train) for est in partial_model.estimators_], axis=0)
            loss_value = mean_squared_error(self.y_train, y_pred)
            self.loss_history.append(loss_value)

        self.plot_loss()

