import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Import DataCleaner for standalone test in __main__ block
# In a real scenario, DataCleaner would be imported and used by main_preprocessing.py
from data_cleaner import DataCleaner
from data_loader import DataLoader # Also need DataLoader for the __main__ block example

class DataTransformer:
    """
    A class to perform data transformation, feature engineering, and imbalance handling.
    """
    def __init__(self):
        """
        Initializes the DataTransformer.
        """
        print("DataTransformer initialized.")
        self.preprocessor_fraud = None
        self.preprocessor_creditcard = None
        self.smote = SMOTE(random_state=42)

    def merge_geolocation_data(self, fraud_df, ip_to_country_df):
        """
        Merges fraud data with IP address to country mapping.

        Args:
            fraud_df (pd.DataFrame): The fraud transactions DataFrame with 'ip_address_int'.
            ip_to_country_df (pd.DataFrame): The IP to country mapping DataFrame
                                              with 'lower_bound_ip_address_int' and 'upper_bound_ip_address_int'.

        Returns:
            pd.DataFrame: The fraud_df with an added 'country' column.
        """
        print("--- Merging Fraud_Data with IpAddress_to_Country ---")

        # Sort ip_to_country for efficient merging
        ip_to_country_sorted = ip_to_country_df.sort_values(by='lower_bound_ip_address_int')
        fraud_df_sorted = fraud_df.sort_values('ip_address_int')

        # Perform a merge_asof to find potential country matches
        merged_fraud_ip = pd.merge_asof(
            fraud_df_sorted,
            ip_to_country_sorted[['lower_bound_ip_address_int', 'upper_bound_ip_address_int', 'country']],
            left_on='ip_address_int',
            right_on='lower_bound_ip_address_int',
            direction='backward', # find the largest lower_bound_ip_address_int less than or equal to ip_address_int
            suffixes=('_fraud', '_ip_range')
        )

        # Filter to ensure the IP address is actually within the upper bound
        merged_fraud_ip['country'] = merged_fraud_ip.apply(
            lambda row: row['country'] if pd.notna(row['ip_address_int']) and \
                                          pd.notna(row['lower_bound_ip_address_int']) and \
                                          pd.notna(row['upper_bound_ip_address_int']) and \
                                          row['ip_address_int'] >= row['lower_bound_ip_address_int'] and \
                                          row['ip_address_int'] <= row['upper_bound_ip_address_int']
                                       else 'Unknown',
            axis=1
        )

        # Drop the temporary merge columns and return
        fraud_df = merged_fraud_ip.drop(columns=['lower_bound_ip_address_int', 'upper_bound_ip_address_int'])
        print(f"Number of unique countries after merge: {fraud_df['country'].nunique()}")

        # Ensure output directory exists and save the merged data
        OUTPUT_DIR = "../data/processed/"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(OUTPUT_DIR, f"merged_fraud_data.csv")
        fraud_df.to_csv(file_path, index=False)
        print(f"Merged fraud data saved to {file_path}")

        return fraud_df

    def engineer_fraud_features(self, fraud_df):
        """
        Engineers time-based and transaction frequency features for fraud_df.

        Args:
            fraud_df (pd.DataFrame): The fraud transactions DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with new engineered features.
        """
        print("--- Feature Engineering for Fraud_Data.csv ---")

        # Time-Based features
        # Ensure 'purchase_time' and 'signup_time' are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(fraud_df['purchase_time']):
            fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
        if not pd.api.types.is_datetime64_any_dtype(fraud_df['signup_time']):
            fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])

        fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
        fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek # Monday=0, Sunday=6
        fraud_df['time_since_signup_seconds'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds()
        fraud_df['time_since_signup_hours'] = fraud_df['time_since_signup_seconds'] / 3600

        # Transaction frequency and velocity
        fraud_df.sort_values(by='purchase_time', inplace=True)

        fraud_df['user_transactions_24h'] = fraud_df.groupby('user_id')['purchase_time'].transform(
            lambda x: x.set_axis(x).rolling('24h', closed='right').count() - 1
        )

        fraud_df['device_transactions_24h'] = fraud_df.groupby('device_id')['purchase_time'].transform(
            lambda x: x.set_axis(x).rolling('24h', closed='right').count() - 1
        )

        fraud_df['ip_transactions_24h'] = fraud_df.groupby('ip_address')['purchase_time'].transform(
            lambda x: x.set_axis(x).rolling('24h', closed='right').count() - 1
        )

        # Fill any potential NaN values from rolling window (e.g., first transaction) with 0
        fraud_df[['user_transactions_24h', 'device_transactions_24h', 'ip_transactions_24h']] = \
            fraud_df[['user_transactions_24h', 'device_transactions_24h', 'ip_transactions_24h']].fillna(0)

        return fraud_df

    def prepare_fraud_data_for_modeling(self, fraud_df):
        """
        Prepares the Fraud_Data.csv for modeling by applying scaling and encoding.

        Args:
            fraud_df (pd.DataFrame): The fraud transactions DataFrame with engineered features.

        Returns:
            tuple: X (features), y (target), preprocessor, and feature names.
        """
        print("\n--- Preparing Fraud_Data.csv for Modeling (Scaling & Encoding) ---")

        # Identify numerical and categorical features
        numerical_features = ['purchase_value', 'age', 'time_since_signup_seconds', 'time_since_signup_hours',
                              'user_transactions_24h', 'device_transactions_24h', 'ip_transactions_24h']
        categorical_features = ['source', 'browser', 'sex', 'hour_of_day', 'day_of_week', 'country']

        # Ensure all identified categorical features are of 'category' dtype for OneHotEncoder
        for col in categorical_features:
            if col in fraud_df.columns:
                fraud_df[col] = fraud_df[col].astype('category')
            else:
                print(f"Warning: Categorical feature '{col}' not found in fraud_df.")

        # Separate features (X) and target (y)
        X = fraud_df.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'ip_address_int', 'class'])
        y = fraud_df['class']

        # Create a preprocessor using ColumnTransformer
        self.preprocessor_fraud = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        return X, y, self.preprocessor_fraud, numerical_features + categorical_features

    def prepare_creditcard_data_for_modeling(self, creditcard_df):
        """
        Prepares the creditcard.csv for modeling by applying scaling.

        Args:
            creditcard_df (pd.DataFrame): The credit card transactions DataFrame.

        Returns:
            tuple: X (features), y (target), preprocessor, and feature names.
        """
        print("\n--- Preparing creditcard.csv for Modeling (Scaling) ---")

        # Identify numerical features (all V1-V28, Time, Amount)
        numerical_features = creditcard_df.drop(columns=['Class']).columns.tolist()

        # Separate features (X) and target (y)
        X = creditcard_df.drop(columns=['Class'])
        y = creditcard_df['Class']

        # Create a preprocessor for creditcard.csv
        self.preprocessor_creditcard = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features)
            ],
            remainder='passthrough'
        )
        return X, y, self.preprocessor_creditcard, numerical_features

    def handle_class_imbalance(self, X_train_processed, y_train, dataset_name=""):
        """
        Applies SMOTE to handle class imbalance in the training data.

        Args:
            X_train_processed (np.array or pd.DataFrame): Preprocessed training features.
            y_train (pd.Series): Training target variable.
            dataset_name (str): Name of the dataset for logging.

        Returns:
            tuple: Resampled X_train and y_train.
        """
        print(f"\n--- Handling Class Imbalance for {dataset_name} (SMOTE) ---")
        print(f"Original training class distribution:\n{y_train.value_counts()}")

        X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train_processed, y_train)

        print(f"Resampled training class distribution:\n{y_train_resampled.value_counts()}")
        print(f"Shape of resampled X_train: {X_train_resampled.shape}")
        return X_train_resampled, y_train_resampled

    def transform_data_for_ml(self, cleaned_fraud_data, cleaned_ip_to_country_data, cleaned_creditcard_data):
        """
        Automates the entire data transformation pipeline for machine learning.

        Args:
            cleaned_fraud_data (pd.DataFrame): Cleaned Fraud_Data.csv DataFrame.
            cleaned_ip_to_country_data (pd.DataFrame): Cleaned IpAddress_to_Country.csv DataFrame.
            cleaned_creditcard_data (pd.DataFrame): Cleaned creditcard.csv DataFrame.

        Returns:
            dict: A dictionary containing the prepared training and testing sets for both datasets.
                  Keys: 'X_train_fraud_resampled', 'y_train_fraud_resampled',
                        'X_test_fraud_processed', 'y_test_fraud',
                        'X_train_creditcard_resampled', 'y_train_creditcard_resampled',
                        'X_test_creditcard_processed', 'y_test_creditcard'
        """
        print("\n--- Starting Automated Data Transformation for ML ---")

        # --- Fraud Data Pipeline ---
        print("\n--- Processing Fraud_Data.csv ---")
        # 1. Merge Geolocation Data
        fraud_data_merged = self.merge_geolocation_data(cleaned_fraud_data.copy(), cleaned_ip_to_country_data.copy())

        # 2. Engineer Features
        fraud_data_engineered = self.engineer_fraud_features(fraud_data_merged.copy())

        # 3. Prepare for Modeling (Scaling & Encoding)
        X_fraud, y_fraud, preprocessor_fraud, fraud_base_features = self.prepare_fraud_data_for_modeling(fraud_data_engineered.copy())

        # Split data into training and testing sets for Fraud_Data.csv
        X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
            X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
        )

        # Create a pipeline for preprocessing (fit on train, transform train/test)
        preprocessing_pipeline_fraud = Pipeline(steps=[('preprocessor', preprocessor_fraud)])
        X_train_fraud_processed = preprocessing_pipeline_fraud.fit_transform(X_train_fraud)
        X_test_fraud_processed = preprocessing_pipeline_fraud.transform(X_test_fraud)

        # 4. Handle Class Imbalance for Fraud_Data.csv
        X_train_fraud_resampled, y_train_fraud_resampled = self.handle_class_imbalance(
            X_train_fraud_processed, y_train_fraud, "Fraud_Data.csv"
        )
        print(f"Fraud Data (Resampled Train) Shape: {X_train_fraud_resampled.shape}, {y_train_fraud_resampled.shape}")
        print(f"Fraud Data (Processed Test) Shape: {X_test_fraud_processed.shape}, {y_test_fraud.shape}")


        # --- Credit Card Data Pipeline ---
        print("\n--- Processing creditcard.csv ---")
        # 1. Prepare for Modeling (Scaling)
        X_creditcard, y_creditcard, preprocessor_creditcard, cc_base_features = self.prepare_creditcard_data_for_modeling(cleaned_creditcard_data.copy())

        # Split data into training and testing sets for creditcard.csv
        X_train_creditcard, X_test_creditcard, y_train_creditcard, y_test_creditcard = train_test_split(
            X_creditcard, y_creditcard, test_size=0.2, random_state=42, stratify=y_creditcard
        )

        # Create a pipeline for preprocessing (fit on train, transform train/test)
        preprocessing_pipeline_creditcard = Pipeline(steps=[('preprocessor', preprocessor_creditcard)])
        X_train_creditcard_processed = preprocessing_pipeline_creditcard.fit_transform(X_train_creditcard)
        X_test_creditcard_processed = preprocessing_pipeline_creditcard.transform(X_test_creditcard)

        # 2. Handle Class Imbalance for creditcard.csv
        X_train_creditcard_resampled, y_train_creditcard_resampled = self.handle_class_imbalance(
            X_train_creditcard_processed, y_train_creditcard, "creditcard.csv"
        )
        print(f"Credit Card Data (Resampled Train) Shape: {X_train_creditcard_resampled.shape}, {y_train_creditcard_resampled.shape}")
        print(f"Credit Card Data (Processed Test) Shape: {X_test_creditcard_processed.shape}, {y_test_creditcard.shape}")


        print("\n--- Automated Data Transformation Complete ---")
        return {
            'X_train_fraud_resampled': X_train_fraud_resampled,
            'y_train_fraud_resampled': y_train_fraud_resampled,
            'X_test_fraud_processed': X_test_fraud_processed,
            'y_test_fraud': y_test_fraud,
            'X_train_creditcard_resampled': X_train_creditcard_resampled,
            'y_train_creditcard_resampled': y_train_creditcard_resampled,
            'X_test_creditcard_processed': X_test_creditcard_processed,
            'y_test_creditcard': y_test_creditcard
        }


if __name__ == '__main__':
    # --- Standalone Example Usage for DataTransformer.transform_data_for_ml ---
    print("--- Running Automated Data Transformation Example ---")

    # 1. Load Data (using DataLoader)
    data_loader = DataLoader()
    dataset_paths = {
        'fraud_data': '../data/processed/fraud_data_cleaned.csv', # Adjust paths if needed for your local setup
        'ip_to_country': '../data/processed/ip_to_country_cleaned.csv',
        'creditcard_data': '../data/processed/creditcard_data_cleaned.csv'
    }
    loaded_dfs = data_loader.load_data(dataset_paths)

    # 2. Clean Data (using DataCleaner)
    data_cleaner = DataCleaner()
    cleaned_dfs = data_cleaner.clean_all_datasets(loaded_dfs)

    fraud_df_cleaned = cleaned_dfs.get('fraud_data')
    ip_df_cleaned = cleaned_dfs.get('ip_to_country')
    cc_df_cleaned = cleaned_dfs.get('creditcard_data')

    if fraud_df_cleaned is None or ip_df_cleaned is None or cc_df_cleaned is None:
        print("Error: Required cleaned data is missing. Cannot proceed with transformation.")
    else:
        # 3. Automate Transformation using DataTransformer
        data_transformer = DataTransformer()
        prepared_data = data_transformer.transform_data_for_ml(
            fraud_df_cleaned,
            ip_df_cleaned,
            cc_df_cleaned
        )

        print("\n--- Prepared Data Overview ---")
        for key, value in prepared_data.items():
            if hasattr(value, 'shape'):
                print(f"{key}: Shape = {value.shape}")
            else:
                print(f"{key}: Type = {type(value)}")

