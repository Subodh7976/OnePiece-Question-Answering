import os 
import sys 
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging 
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer


@dataclass
class TrainPipelineConfig:
    model_save_path: str = os.path.join('artifacts', 'trained_pipe')
    raw_data_path: str = os.path.join('artifacts', 'raw_data')
    clean_data_path: str = os.path.join('artifacts', 'clean_data')
    

class TrainPipeline:
    def __init__(self) -> None:
        self.train_pipeline_config = TrainPipelineConfig()
        self.data_ingestion = DataIngestion(self.train_pipeline_config.raw_data_path,
                                            self.train_pipeline_config.clean_data_path)
        self.model_trainer = ModelTrainer(self.train_pipeline_config.model_save_path)
        
    def train(self, scrape_data: bool = True) -> str:
        '''
        Initializes the training of the model, including scraping the data (if set True)
        Params:
            scrape_data: bool (default = True) - whether to scrape data or not
        Returns:
            Trained model save path: str
        '''
        try:
            logging.info("Model Training started")
            
            if scrape_data:
                self.data_ingestion.initiate_data_ingestion()
            
            model_path = self.model_trainer.initiate_model_trainer(self.train_pipeline_config.clean_data_path)
            
            logging.info("Model Training Completed")
            
            return model_path
            
        except Exception as e:
            raise CustomException(e, sys)
