import os 
import sys 
from dataclasses import dataclass
import pickle

from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import FARMReader
from haystack.nodes import BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_pipe_file_path: str 
    
    
class ModelTrainer:
    def __init__(self, trained_file_path: str):
        '''
        Initializes the Model Trainer class with defined path to save model
        Params:
            trained_file_path: str - File path to (including filename) where to save trained model
        Returns:
            None
        '''
        self.model_trainer_config = ModelTrainerConfig(trained_pipe_file_path=trained_file_path)
        
    def initiate_model_trainer(self, data_path: str):
        '''
        Initializes the Model training on the scraped data, with Roberta Base Squad2 as reader and BM25 as retriever
        Params:
            data_path: str - the path to the data on which model will be indexed
        Returns:
            Path to the saved model: str
        '''
        try:
            logging.info("Model Training initiated")
            
            logging.info("Initializing InMemory Document Store")
            document_store = InMemoryDocumentStore(use_bm25=True)
            
            files_to_index = [os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.txt')]
            
            logging.info("Indexing the document store pipeline")
            indexing_pipeline = TextIndexingPipeline(document_store)
            indexing_pipeline.run_batch(file_paths=files_to_index)
            
            logging.info("Initializing the Reader")
            reader = FARMReader(model_name_or_path='deepset/roberta-base-squad2')
            
            logging.info("Initializing the Retriever")
            retriever = BM25Retriever(document_store=document_store)
            
            logging.info("Initializing the QA Pipeline")
            pipe = ExtractiveQAPipeline(reader, retriever)
            
            logging.info("Saving the QA Pipeline")
            save_object(pipe, self.model_trainer_config.trained_pipe_file_path)
                
            logging.info("Completed the Model Training")
            
            return self.model_trainer_config.trained_pipe_file_path
            
        except Exception as e:
            raise CustomException(e, sys)