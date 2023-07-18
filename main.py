import sys 

from src.pipeline.training import TrainPipeline
from src.logger import logging

SCRAPE_ARG = 'scrape'

arg_list = sys.argv 
train_pipeline = TrainPipeline()

if len(arg_list) > 1:
    if arg_list[1] == SCRAPE_ARG:
        logging.info("Training model with scraping")
        train_pipeline.train()
    else:
        raise Exception(f"Argument {arg_list[1]} is not valid!")
else:
    logging.info("Training model without scraping")
    train_pipeline.train(scrape_data=False)
    
    
    