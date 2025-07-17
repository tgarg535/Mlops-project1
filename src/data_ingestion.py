import logging 
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

log_dir = 'logs'
os.makedirs(log_dir,exist_ok = True)

logger = logging.getLogger(name='data_ingestion')
logger.setLevel(level='DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)






def load_params(param_path:str) -> dict:
    try:
        with open(file=param_path,mode='r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'parameters retrieved successfully from {param_path}')
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', param_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise



def load_data(data_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.debug(f'data loaded from {data_path}')
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise




def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise



def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_path =  os.path.join('./data','raw')
        os.makedirs(raw_data_path)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'))
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'))
        logger.debug(f'train and test data saved successfully to {raw_data_path}')
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(param_path = 'params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_url = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        df = load_data(data_path = data_url)
        final_df = preprocess_data(df = df)
        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state = 1)
        save_data(train_data,test_data,data_path='./data')

    except Exception as e:
        logger.error(f'Failed to complete the data ingestion process: {e}')
        print(f'Eroor:{e}')



if __name__=='__main__':
    main()