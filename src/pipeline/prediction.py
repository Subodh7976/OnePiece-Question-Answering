import sys 
import os 
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, reformat_prediction, check_model_exist


@dataclass
class PredictPipelineConfig:
    saved_model_path: str = os.path.join('artifacts', 'trained_pipe')
    saved_generative_model_path: str = os.path.join('artifacts', 'generative_model')
    saved_generative_tokenizer_path: str = os.path.join('artifacts', 'generative_tokenizer')
    retriever_top_k: int = 10
    reader_top_k: int = 5


class PredictPipeline:
    def __init__(self) -> None:
        self.predict_pipeline_config = PredictPipelineConfig()
        self.prediction_model = None
        self.retriever_params = {"top_k": self.predict_pipeline_config.retriever_top_k}
        self.reader_params = {"top_k": self.predict_pipeline_config.reader_top_k}
        self.prediction_generative_model = None
        self.prediction_generative_tokenizer = None
        self.allocate_model()
    
    def predict(self, query: str) -> dict:
        '''
        predicts the answer with context using the document reader-retriever model
        Params:
            query: str - query to be answered
        Returns:
            answers with context: dict
        '''
        try:
            logging.info("Prediction pipeline started")
            
            prediction = self.prediction_model.run(
                query=query,
                params= {
                    "Retriever": self.retriever_params,
                    "Reader": self.reader_params
                }
            )
            
            prediction = reformat_prediction(prediction)
            
            logging.info("Prediction Completed")
            
            return prediction
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        
    def predict_generative(self, query: str) -> str:
        '''
        predicts the answer with generative model without giving context
        Params:
            query: str - query to be answered
        Returns:
            answer without context: str
        '''
        try:
            logging.info("Generative prediction pipeline started")
            
            tokens = self.prediction_generative_tokenizer.question_encoder(query, return_tensors="pt")['input_ids']
            
            generated = self.prediction_generative_model.generate(tokens)
            predcition = self.prediction_generative_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            
            logging.info("Prediction completed")
            
            return predcition
        except Exception as e:
            logging.exception(e)
            return CustomException(e, sys)
            
    def allocate_model(self):
        if check_model_exist(self.predict_pipeline_config.saved_model_path):
            self.prediction_model = load_object(
                self.predict_pipeline_config.saved_model_path
            )
        
        if check_model_exist(self.predict_pipeline_config.saved_generative_model_path):
            self.prediction_generative_model = load_object(
                self.predict_pipeline_config.saved_generative_model_path
            )
            self.prediction_generative_tokenizer = load_object(
                self.predict_pipeline_config.saved_generative_tokenizer_path
            )
        
        