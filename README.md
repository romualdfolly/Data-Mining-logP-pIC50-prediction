# LogP and pIC50 Prediction

This project aims to predict two important molecular properties: **logP** (logarithm of the octanol/water partition coefficient) and **pIC50** (logarithm of the inverse of the median inhibitory concentration). We use different regression techniques to accomplish this task, leveraging a chemical dataset of molecules.

## Project Structure

### Main Files

- **`DataProcessor.py`**: Contains the `DataProcessor` class for loading, describing, cleaning, and transforming the data. It also handles the extraction of molecular descriptors from SMILES representations.
- **`models/`**: This folder contains the regression models we have implemented to predict **logP** and **pIC50**.
  - **`LR.py`**: Linear Regression.
  - **`MLPR.py`**: Multi-layer Perceptron (Neural Network).
  - **`SVR.py`**: Support Vector Regression (SVM).
  - **`RFR.py`**: Random Forest Regression.
  - **`GBR.py`**: Gradient Boosting Regression.
- **`database/`**: Contains the dataset used for training and testing.

### Features

1. **Data Preprocessing**: The `DataProcessor` class reads and cleans the data, handles missing values, and generates new features from SMILES molecules (e.g., molecular descriptors and atom count).
   
2. **Data Exploration**: You can visualize the distribution of **logP** and **pIC50**, and analyze the impact of molecular descriptors on these values.

3. **Regression Models**: The project implements several regression models to predict **logP** and **pIC50**:
   - Linear Regression
   - Random Forest Regression (RFR)
   - Gradient Boosting Regression (GBR)
   - Support Vector Regression (SVR)
   - Multi-layer Perceptron Regression (MLPR)

4. **Data Standardization**: Before training, the data is standardized to improve model performance.

5. **Data Splitting**: The dataset is split into training, validation, and test sets.

### Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/logP-pIC50-prediction.git
2. pip install -r requirements.txt
   ```bash
   pip install -r requirements.txt```

---
### Data Preprocessing:
Load and prepare your data:

```python
from utils.processor import DataProcessor

path = "path/to/data.csv"
process = DataProcessor(path_to_dataset=path)
process.read_dataset()
process.describe_data()
```

### Train the Models

```python
from models.LR import ProjectLinearRegressor

model = ProjectLinearRegressor()
model.fit(process.X_train_logP, process.y_train_logP)
```

### Evaluate the Models

```python
y_pred = model.predict(process.X_test_logP)
```
### Data Visualization Examples
To visualize the distributions of logP and pIC50, as well as the impact of molecular descriptors

```python
process.visualize_data(about=['MolWt', 'NumHAcceptors'])
```

### Results

