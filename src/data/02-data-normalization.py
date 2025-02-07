import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import click
import logging
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
    input_filepath_train = f'{input_filepath}/X_train.csv'
    input_filepath_test = f'{input_filepath}/X_test.csv'
    output_folderpath = click.prompt('Enter the file path for the output preprocessed data (e.g., output/)', type=click.Path())

    process_data(input_filepath_train, input_filepath_test, output_folderpath)

def process_data(input_filepath_train, input_filepath_test, output_folderpath):
    # Import datasets
    X_train = import_dataset(input_filepath_train, sep=',')
    X_test = import_dataset(input_filepath_test, sep=',')

    # scaling
    scaler = StandardScaler()   
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # transform back to pandas
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Create folder if necessary
    create_folder_if_necessary(output_folderpath)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train_scaled, X_test_scaled, output_folderpath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

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
                print('Invalid response. Please enter \'y\' or \'n\'.')
    else:
        return True

def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()