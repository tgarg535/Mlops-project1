import pandas as pd
import pickle
import os
import logging
import yaml
from sklearn.ensemble import RandomForestClassifier


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel(level='DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel(level="DEBUG")

log_file_path = os.path.join(log_dir,'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(level="DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)





def load_params(params_path:str)->dict:
    try:
        with open(file=params_path,mode='r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'Parameters retrieved from {params_path}')
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise




def load_data(file_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise



def train_model(X_train,y_train,params:dict)->RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same")
        
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        model = RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        model.fit(X_train,y_train)
        logger.debug('Model training completed')
        return model
    
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise




def save_model(model:object,model_path:str)->None:
    try:
        os.makedirs(name=os.path.dirname(model_path),exist_ok=True)
        with open(file=model_path,mode='wb') as file:
            pickle.dump(model,file=file)
        logger.debug('Model saved to %s', model_path)   
     
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise




        

def main():
    try:
        params = load_params(params_path='params.yaml')['model_building']
        train_data = load_data(file_path='./data/processed/train_tfidf.csv')

        X_train = train_data.iloc[:,:-1]
        y_train = train_data.iloc[:,-1]

        model = train_model(X_train,y_train,params)

        model_path = 'models/model.pkl'
        save_model(model=model,model_path=model_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")



if __name__ == '__main__':
    main()