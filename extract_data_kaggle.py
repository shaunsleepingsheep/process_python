import os
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

load_dotenv()

#create raw data folder
current_path = os.getcwd()
raw_data_path = os.path.join(current_path, "raw_data")
os.makedirs(raw_data_path, exist_ok=True)

#download dataset from kagglehub
api = KaggleApi()
api.authenticate()
dataset_name = "tawsifurrahman/covid19-radiography-database"
api.dataset_download_files(
    dataset=dataset_name,
    path=raw_data_path,
    unzip=True
)



