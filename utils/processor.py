import os
import numpy as np
import pandas as pd
from typing import Iterator
from matplotlib import pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import Descriptors
from rdkit.Chem import PeriodicTable

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from collections import Counter




class DataProcessor:
    """
        This class is used to read data, describe data,
        expand features, clean data, split and standardize
    """
    def __init__(self, path_to_dataset:str):
        """ class instances attributes instanciatiation"""
        # check if path is correct
        self.check_existing_dataset(path_to_dataset)

        self.path = path_to_dataset
        self.df = None

        self.cleaned_df_pIC50 = pd.DataFrame()
        self.missing_df_pIC50 = pd.DataFrame()
        self.cleaned_df_logP = pd.DataFrame()
        self.missing_df_logP = pd.DataFrame()

        self.X_train_pIC50 = None
        self.X_test_pIC50 = None
        self.y_train_pIC50 = None
        self.y_test_pIC50 = None
        self.X_validate_pIC50 = None
        self.y_validate_pIC50 = None

        self.X_train_logP = None
        self.X_test_logP = None
        self.y_train_logP = None
        self.y_test_logP = None
        self.X_validate_logP = None
        self.y_validate_logP = None

        self.X_missing_pIC50 = np.empty((0, 0))
        self.X_missing_logP = np.empty((0, 0))

        self.scaler_logP = StandardScaler()
        self.scaler_pIC50 = StandardScaler()
    

    def check_existing_dataset(self, file_path:str):
        """Checks if file exists"""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The dataset is not found at {file_path}")
    

    def read_dataset(self):
        """ This method reads the data from DataSet """
        data = pd.read_csv(self.path)
        self.df = pd.DataFrame(data)
    

    def describe_data(self):
        """ This method describes the data """
        return self.df.describe()
    

    def info_data(self):
        """ This method displays the data main informations on data """
        return self.df.info()
    

    def atoms_cols(self, atoms_source="periodic_table", custum_atoms:Iterator['str'] = []):
        """ This methods returns list. key = atom's symbol; value = 0"""

        # Initialization as periodic table
        periodic_table = rdchem.GetPeriodicTable()

        # Maximal number of atoms in periodic table (118 elements)
        max_atomic_number = 118

        # collection of elements
        all_elements = [periodic_table.GetElementSymbol(i) for i in range(1, max_atomic_number + 1)]
        
        # if user wants custom atoms : not all the periodic table
        if atoms_source == 'custom':
            if len(custum_atoms) == 0:
                raise ValueError("Empty list Exception. In custom mode, the custom_atoms list must contains at least one element")
            
            # intersection to match with periodic table
            all_elements = set(all_elements).intersection(custum_atoms)
        
            # control on new list to check if all provided symbols are atomic symbols
            if len(all_elements) != len(custum_atoms):
                raise ValueError(f"Symbols {set(custum_atoms).difference(all_elements)} are not valid in the Periodic Table")

        # création des colonnes avec valeurs 0 par défaut
        columns = {key: 0 for key in all_elements}
        
        return columns
    

    def get_nb_atoms(self, smiles):
        """This function counts number of each atom the smile chain has"""
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        atom_counts = {}

        # Iterate through atoms in the molecule
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atom_counts:
                atom_counts[symbol] += 1
            else:
                atom_counts[symbol] = 1

        return atom_counts

    
    def get_molecular_descriptors(self, smiles):
        """" this method extract informations based on the provided smile """
        mol = Chem.MolFromSmiles(smiles)
        double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumDoubleBonds': double_bonds,
            'RingCount': Descriptors.RingCount(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol)
        }

        return descriptors
    

    def expand_df_columns(self, atoms_source="periodic_table", custum_atoms:Iterator['str'] = []):
        """ This method is used to expand feature """
        columns_to_add = []

        periodic_table_atoms = self.atoms_cols(atoms_source = atoms_source, custum_atoms = custum_atoms)

        for smiles in self.df["SMILES"]:
            # get descriptors
            descriptors_data = self.get_molecular_descriptors(smiles)
            # periodic atoms copy
            atoms = periodic_table_atoms.copy()
            #count atoms in SMILES
            atoms_occurency = self.get_nb_atoms(smiles)

            # update in periodic
            atoms.update(atoms_occurency)

            # add molecular descriptors to atoms
            atoms.update(descriptors_data)

            columns_to_add.append(atoms.values())
        
        features = atoms.keys()
        df_to_add = pd.DataFrame(data = columns_to_add, columns=features)

        # --->> Remove constant columns
        #df_to_add = df_to_add.loc[:, (df_to_add != df_to_add.iloc[0]).any()]

        # merge DataFrames
        self.df = pd.concat([self.df, df_to_add], axis = 1)
    

    def clean_data(self):
        """Nettoie les données en supprimant les lignes avec pIC50 NaN ou zéro"""
        removed_nan_pIC50 = self.df[self.df['pIC50'].isna()]
        removed_zero_pIC50 = self.df[self.df['pIC50'] == 0]

        if  not removed_nan_pIC50.empty or not removed_zero_pIC50.empty:
            self.missing_df_pIC50 = pd.concat([removed_nan_pIC50, removed_zero_pIC50])
            self.cleaned_df_pIC50 = self.df.drop(self.missing_df_pIC50.index)
        else:
            self.cleaned_df_pIC50 = self.df.copy()

        ###
        removed_nan_logP = self.df[self.df['logP'].isna()]
        if not removed_nan_logP.empty:
            self.missing_df_logP = removed_nan_logP
            self.cleaned_df_logP = self.df.drop(self.missing_df_logP.index)
        else:
            self.cleaned_df_logP = self.df.copy()
    


    def visualize_data(self, about: Iterator[str] = []):
        """ This method plots the target data depends on None or Some main features """
        logP = self.cleaned_df_logP['logP']
        pIC50 = self.cleaned_df_pIC50['pIC50']

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(logP)), logP, c="orange", label="Distribution de logP")
        plt.title("Distribution de logP")
        plt.ylabel("logP")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(range(len(pIC50)), pIC50, c="blue", label="Distribution de pIC50")
        plt.title("Distribution de pIC50")
        plt.ylabel("pIC50")
        plt.legend()

        plt.show()

        for atom in about:
            x_logP = self.cleaned_df_logP[atom]
            x_pIC50 = self.cleaned_df_pIC50[atom]
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.scatter(x_logP, logP, c="orange", label=f"LogP vs {atom}")
            plt.title(f"Distribution de logP en fonction de {atom}")
            plt.xlabel(atom)
            plt.ylabel("logP")
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.scatter(x_pIC50, pIC50, c="blue", label=f"pIC50 vs {atom}")
            plt.title(f"Distribution de pIC50 en fonction de {atom}")
            plt.xlabel(atom)
            plt.ylabel("pIC50")
            plt.legend()
            
            plt.show()

    
    

    def split_data(self, test_size = 0.4, r_state = 42, has_validation = True, percent_on_missing = 0.25):
        """
            This method splits the cleaned datasets into {X, y}_{train, test, val}
            If some feature has missing values, we increase it's train set by adding some percent
        """
        X_logP = self.cleaned_df_logP.drop(columns=['SMILES', 'logP', 'pIC50'])
        y_logP = self.cleaned_df_logP['logP']

        X_pIC50 = self.cleaned_df_pIC50.drop(columns=['SMILES', 'logP', 'pIC50'])
        y_pIC50 = self.cleaned_df_pIC50['pIC50']

        # missing percents computing
        missing_logP_percent = (self.missing_df_logP.shape[0] / self.df.shape[0]) * percent_on_missing
        missing_pIC50_percent = (self.missing_df_pIC50.shape[0] / self.df.shape[0]) * percent_on_missing

        # pIC50 split
        self.X_train_pIC50, X_temp_pIC50, self.y_train_pIC50, y_temp_pIC50 = train_test_split(X_pIC50, y_pIC50, test_size = test_size - missing_pIC50_percent, random_state = r_state)
        if has_validation:
            self.X_val_pIC50, self.X_test_pIC50, self.y_val_pIC50, self.y_test_pIC50 = train_test_split(X_temp_pIC50, y_temp_pIC50, test_size = 0.5, random_state = r_state)
        else:
            self.X_test_pIC50, self.y_test_pIC50 = X_temp_pIC50, y_temp_pIC50

        # logP split
        self.X_train_logP, X_temp_logP, self.y_train_logP, y_temp_logP = train_test_split(X_logP, y_logP, test_size = test_size - missing_logP_percent, random_state = r_state)
        if has_validation:
            self.X_val_logP, self.X_test_logP, self.y_val_logP, self.y_test_logP = train_test_split(X_temp_logP, y_temp_logP, test_size = 0.5, random_state = r_state)
        else:
            self.X_test_logP, self.y_test_logP = X_temp_logP, y_temp_logP


    def validate_splitting(self):
        """ this method ensure all dimensions are correct after spliting """
        print(f'Training set size for pIC50: {self.X_train_pIC50.shape[0]} samples')
        print(f'Validation set size for pIC50: {self.X_val_pIC50.shape[0]} samples')
        print(f'Test set size for pIC50: {self.X_test_pIC50.shape[0]} samples')

        print(f'Training set size for logP: {self.X_train_logP.shape[0]} samples')
        print(f'Validation set size for logP: {self.X_val_logP.shape[0]} samples')
        print(f'Test set size for logP: {self.X_test_logP.shape[0]} samples')

        assert len(set(self.X_train_pIC50.index).intersection(self.X_val_pIC50.index)) == 0
        assert len(set(self.X_train_pIC50.index).intersection(self.X_test_pIC50.index)) == 0
        assert len(set(self.X_val_pIC50.index).intersection(self.X_test_pIC50.index)) == 0

        assert len(set(self.X_train_logP.index).intersection(self.X_val_logP.index)) == 0
        assert len(set(self.X_train_logP.index).intersection(self.X_test_logP.index)) == 0
        assert len(set(self.X_val_logP.index).intersection(self.X_test_logP.index)) == 0


    def standardize_data(self):
        """ This method helps us to standardize the data """
        self.scaler_pIC50 = StandardScaler()
        self.X_train_pIC50 = self.scaler_pIC50.fit_transform(self.X_train_pIC50)
        self.X_val_pIC50 = self.scaler_pIC50.transform(self.X_val_pIC50)
        self.X_test_pIC50 = self.scaler_pIC50.transform(self.X_test_pIC50)

        scaler_logP = StandardScaler()
        self.X_train_logP = self.scaler_logP.fit_transform(self.X_train_logP)
        self.X_val_logP = self.scaler_logP.transform(self.X_val_logP)
        self.X_test_logP = self.scaler_logP.transform(self.X_test_logP)


    def validate_scaling(self):
        # Check the mean and standard deviation of the scaled features for pIC50
        scaled_means_pIC50 = self.X_train_pIC50.mean(axis=0)
        scaled_stds_pIC50 = self.X_train_pIC50.std(axis=0)

        print("Means of scaled features for pIC50:", scaled_means_pIC50)
        print("Standard deviations of scaled features for pIC50:", scaled_stds_pIC50)

        # Check the mean and standard deviation of the scaled features for logP
        scaled_means_logP = self.X_train_logP.mean(axis=0)
        scaled_stds_logP = self.X_train_logP.std(axis=0)

        print("Means of scaled features for logP:", scaled_means_logP)
        print("Standard deviations of scaled features for logP:", scaled_stds_logP)
    

    def prepare_missing_for_predict(self):
        """ This method prepares raws that contain missing target for prediction """
        if self.missing_df_pIC50.shape[0] != 0:  # if we have missing pIC50 values in he original DataSet
            self.X_missing_pIC50 = self.missing_df_pIC50.drop(columns=['SMILES', 'logP', 'pIC50'])          # features separation
            self.X_missing_pIC50 = self.scaler_pIC50.transform(self.X_missing_pIC50)                        # scaling
        
        if self.missing_df_logP.shape[0] != 0:  # if we have missing logP values in he original DataSet
            self.X_missing_logP = self.missing_df_logP.drop(columns=['SMILES', 'logP', 'pIC50'])            # features separation
            self.X_missing_logP = self.scaler_logP.transform(self.X_missing_logP)                           # scaling
    

    def get_pIC50_splited_data_copy(self):
        """ return the copy of pIC50 splited data """
        return self.X_train_pIC50.copy(), self.X_test_pIC50.copy(), self.X_val_pIC50.copy(), self.y_train_pIC50.copy(), self.y_test_pIC50.copy(), self.y_val_pIC50.copy()
    

    def get_logP_splited_data_copy(self):
        """ return the copy of logP splited data """
        return self.X_train_logP.copy(), self.X_test_logP.copy(), self.X_val_logP.copy(), self.y_train_logP.copy(), self.y_test_logP.copy(), self.y_val_logP.copy()
    