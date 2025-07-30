import pandas as pd

class DataLoader:
    """
    A class to load the required datasets for the fraud detection project.
    """
    def __init__(self):
        """
        Initializes the DataLoader.
        """
        print("DataLoader initialized.")

    def load_data(self, dataset_paths_dict):
        """
        Loads multiple datasets from CSV files based on a dictionary of paths.

        Args:
            dataset_paths_dict (dict): A dictionary where keys are descriptive names
                                       for the datasets (e.g., 'fraud_data', 'ip_to_country')
                                       and values are their respective file paths.
                                       Example:
                                       {
                                           'fraud_data': 'Fraud_Data.csv',
                                           'ip_to_country': 'IpAddress_to_Country.csv',
                                           'creditcard_data': 'creditcard.csv'
                                       }

        Returns:
            dict: A dictionary where keys are the dataset names and values are
                  the loaded pandas DataFrames. If a file is not found, its
                  corresponding value in the dictionary will be None.
        """
        print("--- Loading Data ---")
        loaded_data = {}
        for name, path in dataset_paths_dict.items():
            try:
                df = pd.read_csv(path)
                loaded_data[name] = df
                print(f"Dataset '{name}' loaded successfully from '{path}'.")
            except FileNotFoundError:
                print(f"Error: File not found for '{name}' at '{path}'. Setting to None.")
                loaded_data[name] = None
            except Exception as e:
                print(f"An unexpected error occurred while loading '{name}' from '{path}': {e}. Setting to None.")
                loaded_data[name] = None
        return loaded_data

if __name__ == '__main__':
    # Example usage:
    loader = DataLoader()

    # Define the paths for the datasets in a dictionary
    paths = {
        'fraud_data': '../data/raw/Fraud_Data.csv',
        'ip_to_country': '../data/raw/IpAddress_to_Country.csv',
        'creditcard_data': '../data/raw/creditcard.csv'
    }

    # Load the data using the new method
    loaded_dfs = loader.load_data(paths)

    # Access the loaded DataFrames from the dictionary
    fraud_df = loaded_dfs.get('fraud_data')
    ip_df = loaded_dfs.get('ip_to_country')
    cc_df = loaded_dfs.get('creditcard_data')

    if fraud_df is not None:
        print("\nFraud Data Head:")
        print(fraud_df.head())
    else:
        print("\nFraud Data was not loaded.")

    if ip_df is not None:
        print("\nIP to Country Data Head:")
        print(ip_df.head())
    else:
        print("\nIP to Country Data was not loaded.")

    if cc_df is not None:
        print("\nCredit Card Data Head:")
        print(cc_df.head())
    else:
        print("\nCredit Card Data was not loaded.")
