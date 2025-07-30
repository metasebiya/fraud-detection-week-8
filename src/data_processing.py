import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_transformer import DataTransformer

def perform_eda(df, df_name):
    """
    Performs basic Exploratory Data Analysis (EDA) on a DataFrame.
    Includes descriptive statistics, class distribution, and some visualizations.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        df_name (str): The name of the DataFrame for logging and plot titles.
    """
    print(f"\n--- Exploratory Data Analysis (EDA) for {df_name} ---")
    print("\nDescriptive Statistics for numerical features:")
    print(df.describe())

    if 'class' in df.columns or 'Class' in df.columns:
        target_col = 'class' if 'class' in df.columns else 'Class'
        print(f"\nClass Distribution in {df_name}:")
        print(df[target_col].value_counts())
        print(f"Fraudulent transactions: {df[target_col].value_counts()[1]} ({df[target_col].value_counts(normalize=True)[1]:.2%})")

        plt.figure(figsize=(6, 4))
        sns.countplot(x=target_col, data=df)
        plt.title(f'Class Distribution in {df_name}')
        plt.xlabel(f'{target_col} (0: Non-Fraud, 1: Fraud)')
        plt.ylabel('Number of Transactions')
        plt.show()

        # Bivariate analysis for Fraud_Data.csv specific categorical features
        if df_name == 'Fraud_Data.csv':
            categorical_cols = ['source', 'browser', 'sex']
            for col in categorical_cols:
                if col in df.columns:
                    fraud_rate = df.groupby(col)[target_col].mean().sort_values(ascending=False)
                    print(f"\nFraud rate by {col} in {df_name}:\n{fraud_rate}")
                    plt.figure(figsize=(8, 5))
                    sns.barplot(x=col, y=target_col, data=df)
                    plt.title(f'Fraud Rate by {col} in {df_name}')
                    plt.ylabel('Fraud Rate')
                    plt.show()

            # Distribution of numerical features by class
            numerical_cols = ['purchase_value', 'age']
            for col in numerical_cols:
                if col in df.columns:
                    plt.figure(figsize=(12, 5))
                    sns.histplot(data=df, x=col, hue=target_col, kde=True, bins=50)
                    plt.title(f'{col} Distribution by Class in {df_name}')
                    plt.show()
        elif df_name == 'creditcard.csv':
            if 'Amount' in df.columns:
                plt.figure(figsize=(12, 5))
                sns.histplot(data=df, x='Amount', hue=target_col, kde=True, bins=50)
                plt.title(f'Transaction Amount Distribution by Class in {df_name}')
                plt.show()

def main():
    """
    Main function to orchestrate the data loading, cleaning, transformation,
    and preprocessing steps for the fraud detection project.
    """
    # Initialize classes
    data_loader = DataLoader()
    data_cleaner = DataCleaner()
    data_transformer = DataTransformer()

    # --- 1. Load Data ---
    fraud_data, ip_to_country, creditcard_data = data_loader.load_data()

    if fraud_data is None or ip_to_country is None or creditcard_data is None:
        print("Exiting due to data loading errors.")
        return

    print("\n--- Initial Data Inspection and Cleaning ---")
    # --- 2. Handle Missing Values & Data Cleaning ---

    # Fraud_Data.csv
    print("\nFraud_Data.csv Info (Before Cleaning):")
    fraud_data.info()
    print("\nMissing values in Fraud_Data.csv (Before Cleaning):")
    print(fraud_data.isnull().sum())
    fraud_data = data_cleaner.handle_missing_values(fraud_data, 'ip_address', strategy='drop')
    fraud_data = data_cleaner.convert_time_columns(fraud_data, ['signup_time', 'purchase_time'])
    fraud_data = data_cleaner.remove_duplicates(fraud_data, "Fraud_Data.csv")

    # IpAddress_to_Country.csv
    print("\nIpAddress_to_Country.csv Info (Before Cleaning):")
    ip_to_country.info()
    print("\nMissing values in IpAddress_to_Country.csv (Before Cleaning):")
    print(ip_to_country.isnull().sum())
    ip_to_country = data_cleaner.remove_duplicates(ip_to_country, "IpAddress_to_Country.csv")

    # creditcard.csv
    print("\ncreditcard.csv Info (Before Cleaning):")
    creditcard_data.info()
    print("\nMissing values in creditcard.csv (Before Cleaning):")
    print(creditcard_data.isnull().sum())
    creditcard_data = data_cleaner.remove_duplicates(creditcard_data, "creditcard.csv")

    # --- 3. Exploratory Data Analysis (EDA) ---
    perform_eda(fraud_data.copy(), 'Fraud_Data.csv') # Use .copy() to avoid SettingWithCopyWarning if EDA modifies
    perform_eda(creditcard_data.copy(), 'creditcard.csv')

    # --- 4. Merge Datasets for Geolocation Analysis ---
    # Convert IPs to integers before merging
    fraud_data = data_cleaner.convert_ip_to_int_columns(fraud_data, 'ip_address', 'ip_address_int')
    ip_to_country = data_cleaner.convert_ip_to_int_columns(ip_to_country, 'lower_bound_ip_address', 'lower_bound_ip_address_int')
    ip_to_country = data_cleaner.convert_ip_to_int_columns(ip_to_country, 'upper_bound_ip_address', 'upper_bound_ip_address_int')

    fraud_data = data_transformer.merge_geolocation_data(fraud_data, ip_to_country)
    print("\nFraud_Data.csv after merging with IP addresses (head):")
    print(fraud_data.head())
    print(f"Top 10 countries:\n{fraud_data['country'].value_counts().head(10)}")

    # --- 5. Feature Engineering (Fraud_Data.csv) ---
    fraud_data = data_transformer.engineer_fraud_features(fraud_data)
    print("\nFraud_Data.csv with new engineered features (head):")
    print(fraud_data[['purchase_time', 'signup_time', 'time_since_signup_hours',
                      'hour_of_day', 'day_of_week', 'user_transactions_24h',
                      'device_transactions_24h', 'ip_transactions_24h', 'class']].head())

    # --- 6. Data Transformation ---

    # Prepare Fraud Data for Modeling
    X_fraud, y_fraud, preprocessor_fraud, fraud_base_features = data_transformer.prepare_fraud_data_for_modeling(fraud_data)

    # Split data into training and testing sets for Fraud_Data.csv
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
        X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
    )

    # Create a pipeline for preprocessing
    preprocessing_pipeline_fraud = Pipeline(steps=[('preprocessor', preprocessor_fraud)])

    # Apply preprocessing to training data *before* SMOTE
    X_train_fraud_processed = preprocessing_pipeline_fraud.fit_transform(X_train_fraud)
    X_test_fraud_processed = preprocessing_pipeline_fraud.transform(X_test_fraud)

    # Get feature names after one-hot encoding for Fraud Data
    ohe_feature_names_fraud = preprocessor_fraud.named_transformers_['cat'].get_feature_names_out(
        [col for col in fraud_base_features if col in ['source', 'browser', 'sex', 'hour_of_day', 'day_of_week', 'country']]
    )
    all_fraud_feature_names = [col for col in fraud_base_features if col not in ['source', 'browser', 'sex', 'hour_of_day', 'day_of_week', 'country']] + list(ohe_feature_names_fraud)

    # Handle Class Imbalance for Fraud_Data.csv
    X_train_fraud_resampled, y_train_fraud_resampled = data_transformer.handle_class_imbalance(
        X_train_fraud_processed, y_train_fraud, "Fraud_Data.csv"
    )

    # Prepare Credit Card Data for Modeling
    X_creditcard, y_creditcard, preprocessor_creditcard, cc_base_features = data_transformer.prepare_creditcard_data_for_modeling(creditcard_data)

    # Split data into training and testing sets for creditcard.csv
    X_train_creditcard, X_test_creditcard, y_train_creditcard, y_test_creditcard = train_test_split(
        X_creditcard, y_creditcard, test_size=0.2, random_state=42, stratify=y_creditcard
    )

    # Create a pipeline for preprocessing
    preprocessing_pipeline_creditcard = Pipeline(steps=[('preprocessor', preprocessor_creditcard)])

    # Apply preprocessing to training data *before* SMOTE
    X_train_creditcard_processed = preprocessing_pipeline_creditcard.fit_transform(X_train_creditcard)
    X_test_creditcard_processed = preprocessing_pipeline_creditcard.transform(X_test_creditcard)

    # Handle Class Imbalance for creditcard.csv
    X_train_creditcard_resampled, y_train_creditcard_resampled = data_transformer.handle_class_imbalance(
        X_train_creditcard_processed, y_train_creditcard, "creditcard.csv"
    )

    print("\n--- Preprocessing Complete ---")
    print("You now have preprocessed and resampled training data (X_train_fraud_resampled, y_train_fraud_resampled) for Fraud_Data.csv")
    print("and (X_train_creditcard_resampled, y_train_creditcard_resampled) for creditcard.csv, ready for model training.")
    print("The corresponding test sets (X_test_fraud_processed, y_test_fraud) and (X_test_creditcard_processed, y_test_creditcard) are also ready for evaluation.")

if __name__ == '__main__':
    main()
