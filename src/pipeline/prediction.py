import sys 
import os 
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, reformat_prediction


@dataclass
class PredictPipelineConfig:
    saved_model_path: str = os.path.join('artifacts', 'trained_pipe')
    retriever_top_k: int = 10
    reader_top_k: int = 5


class PredictPipeline:
    def __init__(self) -> None:
        self.predict_pipeline_config = PredictPipelineConfig()
        self.prediction_model = load_object(self.predict_pipeline_config.saved_model_path)
        self.retriever_params = {"top_k": self.predict_pipeline_config.retriever_top_k}
        self.reader_params = {"top_k": self.predict_pipeline_config.reader_top_k}
    
    def predict(self, query: str) -> dict:
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
            raise CustomException(e, sys)
            