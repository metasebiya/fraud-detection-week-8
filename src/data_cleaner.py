import pandas as pd
import numpy as np
from data_loader import DataLoader
import os
import socket # Import the socket module for IP conversion
import struct # Import the struct module for byte packing/unpacking

class DataCleaner:
    """
    A class to perform data cleaning operations on the datasets.
    """
    def __init__(self):
        """
        Initializes the DataCleaner.
        """
        print("DataCleaner initialized.")

    def handle_missing_values(self, df, column_name, strategy='drop'):
        """
        Handles missing values in a specified column of a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            column_name (str): The name of the column to check for missing values.
            strategy (str): The strategy to handle missing values ('drop' or 'impute').
                            Currently, only 'drop' is implemented for 'ip_address'.

        Returns:
            pd.DataFrame: The DataFrame after handling missing values.
        """
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' not found in DataFrame.")
            return df

        missing_count = df[column_name].isnull().sum()
        if missing_count > 0:
            print(f"Handling {missing_count} missing values in '{column_name}' using '{strategy}' strategy.")
            if strategy == 'drop':
                initial_rows = df.shape[0]
                df.dropna(subset=[column_name], inplace=True)
                print(f"Dropped {initial_rows - df.shape[0]} rows with missing '{column_name}'.")
            elif strategy == 'impute':
                # Placeholder for imputation logic if needed in the future
                print(f"Imputation for '{column_name}' is not yet implemented. Skipping.")
        else:
            print(f"No missing values found in '{column_name}'.")
        return df

    def convert_time_columns(self, df, time_columns):
        """
        Converts specified columns in a DataFrame to datetime objects.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            time_columns (list): A list of column names to convert to datetime.

        Returns:
            pd.DataFrame: The DataFrame with converted time columns.
        """
        print(f"Converting columns {time_columns} to datetime objects.")
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
            else:
                print(f"Warning: Time column '{col}' not found in DataFrame.")
        return df

    def remove_duplicates(self, df, df_name="DataFrame"):
        """
        Removes duplicate rows from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            df_name (str): A name for the DataFrame for logging purposes.

        Returns:
            pd.DataFrame: The DataFrame with duplicate rows removed.
        """
        initial_rows = df.shape[0]
        df.drop_duplicates(inplace=True)
        dropped_count = initial_rows - df.shape[0]
        if dropped_count > 0:
            print(f"Removed {dropped_count} duplicate rows from {df_name}.")
        else:
            print(f"No duplicate rows found in {df_name}.")
        return df

    def ip_to_int(self, ip_str):
        """
        Converts an IP address string to its integer representation.
        Prioritizes standard IPv4 dotted-decimal format (A.B.C.D) using socket/struct.
        Falls back to float-like string conversion (e.g., '12345.678') if IPv4 parsing fails.

        Args:
            ip_str (str): The IP address string.

        Returns:
            int: The integer representation of the IP address, or np.nan if input is NaN or invalid format.
        """
        if pd.isna(ip_str):
            return np.nan

        try:
            # Attempt to parse as standard IPv4 dotted-decimal format (A.B.C.D)
            # socket.inet_aton converts IP string to 32-bit packed binary format
            # struct.unpack('>I', ...) unpacks 4 bytes as a single unsigned integer (big-endian)
            return struct.unpack('>I', socket.inet_aton(ip_str))[0]
        except OSError:
            # If socket.inet_aton fails (e.g., invalid IPv4 format), try to parse as a float then int
            try:
                # This handles cases like '732758368.79972' by converting to float then int
                return int(float(ip_str))
            except ValueError:
                print(f"Warning: Could not convert IP '{ip_str}' to integer (neither IPv4 nor numeric). Returning NaN.")
                return np.nan
            except Exception as e:
                print(f"An unexpected error occurred converting IP '{ip_str}' as numeric: {e}. Returning NaN.")
                return np.nan
        except Exception as e:
            print(f"An unexpected error occurred converting IP '{ip_str}' as IPv4: {e}. Returning NaN.")
            return np.nan

    def int_to_ip(self, ip_int):
        """
        Converts an integer representation of an IP address back to its dotted-decimal string format.

        Args:
            ip_int (int): The integer representation of the IP address.

        Returns:
            str: The dotted-decimal string representation of the IP address, or None if input is invalid.
        """
        if pd.isna(ip_int):
            return None
        try:
            # struct.pack('>I', ...) packs the integer into 4 bytes (big-endian)
            # socket.inet_ntoa converts 32-bit packed binary format to IP string
            return socket.inet_ntoa(struct.pack('>I', int(ip_int)))
        except (struct.error, OSError, ValueError) as e:
            print(f"Warning: Could not convert integer '{ip_int}' to IP string: {e}. Returning None.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred converting integer '{ip_int}' to IP string: {e}. Returning None.")
            return None

    def convert_ip_to_int_columns(self, df, ip_column, new_int_column):
        """
        Applies the IP to integer conversion to a specified column in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            ip_column (str): The name of the column containing IP address strings.
            new_int_column (str): The name for the new column with integer IPs.

        Returns:
            pd.DataFrame: The DataFrame with the new integer IP column.
        """
        if ip_column not in df.columns:
            print(f"Warning: IP column '{ip_column}' not found in DataFrame. Skipping IP conversion.")
            return df
        print(f"Converting IP addresses in '{ip_column}' to integer format.")
        # Added .astype(str) to ensure that apply is always called on string representations
        print(f"Converting IP addresses in '{ip_column}' ")
        df[ip_column] = df[ip_column].astype(float).astype(int)
        df[new_int_column] = df[ip_column].astype(str).apply(self.ip_to_int)
        return df

    def clean_all_datasets(self, datasets_dict):
        """
        Performs cleaning operations on all DataFrames provided in the dictionary.

        Args:
            datasets_dict (dict): A dictionary of DataFrames, where keys are dataset names.

        Returns:
            dict: A dictionary of cleaned DataFrames.
        """
        print("\n--- Performing Cleaning for All Datasets ---")
        cleaned_datasets = {}

        # Clean Fraud_Data.csv
        if 'fraud_data' in datasets_dict and datasets_dict['fraud_data'] is not None:
            print("\n--- Cleaning Fraud_Data.csv ---")
            df_fraud = datasets_dict['fraud_data'].copy() # Work on a copy
            df_fraud = self.handle_missing_values(df_fraud, 'ip_address', strategy='drop')
            df_fraud = self.convert_time_columns(df_fraud, ['signup_time', 'purchase_time'])
            df_fraud = self.remove_duplicates(df_fraud, "Fraud_Data.csv")
            df_fraud = self.convert_ip_to_int_columns(df_fraud, 'ip_address', 'ip_address_int')
            cleaned_datasets['fraud_data'] = df_fraud

        else:
            print("Fraud_Data.csv not found or is None in the input dictionary. Skipping cleaning.")
            cleaned_datasets['fraud_data'] = None

        # Clean IpAddress_to_Country.csv
        if 'ip_to_country' in datasets_dict and datasets_dict['ip_to_country'] is not None:
            print("\n--- Cleaning IpAddress_to_Country.csv ---")
            df_ip = datasets_dict['ip_to_country'].copy() # Work on a copy
            df_ip = self.remove_duplicates(df_ip, "IpAddress_to_Country.csv")
            df_ip = self.convert_ip_to_int_columns(df_ip, 'lower_bound_ip_address', 'lower_bound_ip_address_int')
            df_ip = self.convert_ip_to_int_columns(df_ip, 'upper_bound_ip_address', 'upper_bound_ip_address_int')
            cleaned_datasets['ip_to_country'] = df_ip
        else:
            print("IpAddress_to_Country.csv not found or is None in the input dictionary. Skipping cleaning.")
            cleaned_datasets['ip_to_country'] = None

        # Clean creditcard.csv
        if 'creditcard_data' in datasets_dict and datasets_dict['creditcard_data'] is not None:
            print("\n--- Cleaning creditcard.csv ---")
            df_cc = datasets_dict['creditcard_data'].copy() # Work on a copy
            df_cc = self.remove_duplicates(df_cc, "creditcard.csv")
            cleaned_datasets['creditcard_data'] = df_cc
        else:
            print("creditcard.csv not found or is None in the input dictionary. Skipping cleaning.")
            cleaned_datasets['creditcard_data'] = None

        print("\n--- All Datasets Cleaned ---")
        return cleaned_datasets


if __name__ == '__main__':
    # Example Usage:


    path = {
        'fraud_data': '../data/raw/Fraud_Data.csv',
        'ip_to_country': '../data/raw/IpAddress_to_Country.csv',
        'creditcard_data': '../data/raw/creditcard.csv'
    }
    loader = DataLoader()
    datasets_to_clean = loader.load_data(path)

    cleaner = DataCleaner()
    cleaned_dfs = cleaner.clean_all_datasets(datasets_to_clean)

    OUTPUT_DIR = "../data/processed/"
    for name, item in cleaned_dfs.items():
        if item is not None:
            file_path = os.path.join(OUTPUT_DIR, f"{name}_cleaned.csv")
            item.to_csv(file_path, index=False)  # index=False to avoid writing DataFrame index as a column
            print(f"Saved cleaned '{name}' to {file_path}")
        else:
            print(f"Skipping saving for '{name}' as it was not loaded or cleaned successfully.")

