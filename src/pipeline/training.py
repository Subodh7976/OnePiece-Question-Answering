import os 
import sys 
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging 
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.components.generative_trainer import GenerativeModel


@dataclass
class TrainPipelineConfig:
    model_save_path: str = os.path.join('artifacts', 'trained_pipe')
    generative_model_save_path: str = os.path.join('artifacts', 'generative_model')
    generative_tokenizer_save_path: str = os.path.join('artifacts', 'generative_tokenizer')
    raw_data_path: str = os.path.join('artifacts', 'raw_data')
    clean_data_path: str = os.path.join('artifacts', 'clean_data')
    

class TrainPipeline:
    def __init__(self) -> None:
        self.train_pipeline_config = TrainPipelineConfig()
        self.data_ingestion = DataIngestion(self.train_pipeline_config.raw_data_path,
                                            self.train_pipeline_config.clean_data_path)
        self.model_trainer = ModelTrainer(self.train_pipeline_config.model_save_path)
        self.generative_trainer = GenerativeModel(
            self.train_pipeline_config.generative_model_save_path,
            self.train_pipeline_config.generative_tokenizer_save_path
        )
        
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
            logging.exception(e)
            raise CustomException(e, sys)

    def train_generative(self, scrape_data: bool = True) -> str:
        '''
        Initializes the training of the generative model, including scraping the data (is set True)
        Params:
            scrape_data: bool (default = True) - whether to scrape the data or not
        Returns:
            Trained model save path: str
        '''
        try:
            logging.info("Generative Model Training started")
            
            if scrape_data:
                self.data_ingestion.initiate_data_ingestion()
                
            generative_model_path = self.generative_trainer.initiate_generative_trainer(
                self.train_pipeline_config.clean_data_path
            )
            
            logging.info("Generative Model Completed")
            return generative_model_path
            
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    