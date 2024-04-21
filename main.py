import sys 

from src.pipeline.training import TrainPipeline
from src.logger import logging


SCRAPE_ARG = 'scrape'
GEN_ARG = 'gen'

arg_list = sys.argv 
train_pipeline = TrainPipeline()

if GEN_ARG in arg_list:
    if SCRAPE_ARG in arg_list:
        logging.info("Training generative model with scraping")
        train_pipeline.train_generative()
    else:
        logging.info("Training generative model without scraping")
        train_pipeline.train_generative(scrape_data=False)
else:
    if SCRAPE_ARG in arg_list:
        logging.info("Training model with scraping")
        train_pipeline.train()
    else:
        logging.info("Training model without scraping")
        train_pipeline.train(scrape_data=False)
    
    
    
