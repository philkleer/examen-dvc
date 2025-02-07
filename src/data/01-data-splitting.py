import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    ''' Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    '''
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    input_filepath_data = f'{input_filepath}/raw.csv'
    output_folderpath = click.prompt('Enter the file path for the output preprocessed data (e.g., output/preprocessed_data.csv)', type=click.Path())

    # Create folder if necessary
    create_folder_if_necessary(output_folderpath)

    process_data(input_filepath_data, output_folderpath)

def process_data(input_filepath, output_folderpath):
    # Import datasets
    df = import_dataset(input_filepath, sep=',')

    df = df.drop('date', axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train, X_test, y_train, y_test, output_folderpath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def check_existing_file(file_path):
    '''Check if a file already exists. If it does, ask if we want to overwrite it.'''
    if os.path.isfile(file_path):
        while True:
            response = input(f'File {os.path.basename(file_path)} already exists. Do you want to overwrite it? (y/n): ')
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print('Invalid response. Please enter "y" or "n".')
    else:
        return True

def check_existing_folder(folder_path):
    '''Check if a folder already exists. If it doesn't, ask if we want to create it.'''
    if not os.path.exists(folder_path):
        while True:
            response = input(f'{os.path.basename(folder_path)} doesn\'t exists. Do you want to create it? (y/n): ')
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print('Invalid response. Please enter \'y\' or \'n\'.')
    else:
        return False

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):

    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    main()