import os 
import sys
from dataclasses import dataclass
from urllib.parse import unquote
import math 
import numpy as np 
import pandas as pd 
from datasets import Dataset

import faiss 
from transformers import (
    DPRContextEncoderTokenizerFast,
    DPRContextEncoder,
    RagRetriever,
    RagTokenizer,
    RagSequenceForGeneration
)
import torch 

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class GenerativeModelConfig:
    generative_model_save_path: str 
    generative_tokenizer_save_path: str
    passage_length: int = 150
    batch_size: int = 16
    dpr_pretrained_model: str = "facebook/dpr-ctx_encoder-multiset-base"
    rag_pretrained_model: str = "facebook/rag-sequence-nq"
    index_dim: int = 768
    index_m: int = 128
    

class GenerativeModel:
    def __init__(self, model_save_path: str, tokenizer_save_path: str) -> None:
        '''
        Create an instance for Generative Model Training class.
        Params:
            model_save_path: str - the path to save model
            tokenizer_save_path: str - the path to save tokenizer
        Returns:
            None
        '''
        self.generative_model_config = GenerativeModelConfig(
            model_save_path,
            tokenizer_save_path
        )
        self.device = None
        
    def initiate_generative_trainer(self, data_path: str):
        '''
        Initiates the model training for Generative Model based on RAG model.
        Params:
            data_path: str - the path for the training data
        Returns:
            Path of the saved model: str
            Path of the saved tokenizer: str
        '''
        
        try:
            logging.info("Generative model training initiated.")
            
            titles, articles = self.__get_titles_articles(data_path)
            passage_titles, passages = self.__generate_passages(titles, articles)
            
            chunked_corpus = {"title": passage_titles, "text": passages}
            
            logging.info("Tokenizing the context passages")
            outputs = self.__generate_tokens(chunked_corpus)
            
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logging.info("The GPU used is {}".format(self.device))
            
            logging.info("Processing embeddings for the generated tokens")
            embeddings = self.__generate_embeddings(outputs['input_ids'])
            
            embeddings = np.concatenate(embeddings, axis=0)
            
            dataset = pd.DataFrame(chunked_corpus)
            dataset = Dataset.from_pandas(dataset)
            
            embs = []
            for i in range(embeddings.shape[0]):
                embs.append(embeddings[i, :])
                
            dataset = dataset.add_column('embeddings', embs)
            
            logging.info("Indexing the dataset")
            dataset = self.__generate_indexed_dataset(dataset)
            
            logging.info("Preparing retriever")
            rag_retriever = RagRetriever.from_pretrained(
                self.generative_model_config.rag_pretrained_model,
                use_dummy_dataset=False,
                indexed_dataset=dataset,
                index_name='embeddings'
            )
            
            logging.info("Preparing tokenizer")
            rag_tokenizer = RagTokenizer.from_pretrained(
                self.generative_model_config.rag_pretrained_model
            )
            
            logging.info("Preparing model")
            rag_model = RagSequenceForGeneration.from_pretrained(
                self.generative_model_config.rag_pretrained_model,
                retriever=rag_retriever
            )
            
            logging.info("Saving the model and tokenizer")
            save_object(
                rag_model,
                self.generative_model_config.generative_model_save_path
            )
            save_object(
                rag_tokenizer,
                self.generative_model_config.generative_tokenizer_save_path
            )
            
            logging.info("Generative model training completed")
            
            return (
                self.generative_model_config.generative_model_save_path,
                self.generative_model_config.generative_tokenizer_save_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def __get_titles_articles(self, data_path: str) -> tuple(list, list):
        '''
        retrieves the titles and articles from the defined data path, and returns in the form of list
        Params:
            data_path: str - the path for the data location
        Returns:
            Tuple consisting two list for titles and articles: tuple(list, list)
        '''
        titles = []
        articles = []
            
        for filename in os.listdir(data_path):
            if not filename.endswith('.txt'):
                continue
                    
            with open(os.path.join(data_path, filename), 'rb') as file:
                title = unquote(filename[:-4])
                    
                if len(title) == 0 or len(title.strip()) == 0:
                    continue
                        
                titles.append(title)
                    
                articles.append(file.read().decode('utf-8'))
                
                    
        return (titles, articles)
        
    def __generate_passages(self, titles: list[str], articles: list[str]) -> tuple(list[str], list[str]):
        '''
        generates the passages for the given titles and articles lists
        Params:
            titles: list - list containing all the titles
            articles: list - list containing all the articles
        Returns:
            tuple containing list for passage titles and passages: tuple(list, list)
        '''    
        passage_titles = []
        passages = []
        
        for i in range(len(titles)):
            title = titles[i]
            article = articles[i]
            
            if len(article) == 0:
                continue
                
            words = article.split()
            
            for j in range(0, len(words), self.generative_model_config.passage_length):
                chunk_words = words[j:j+self.generative_model_config.passage_length]
                chunk = " ".join(chunk_words)
                chunk = chunk.strip()
                
                if len(chunk) == 0:
                    continue
                
                passage_titles.append(title)
                passages.append(chunk)
                
        return (passage_titles, passages)
    
    
    def __generate_tokens(self, corpus: dict) -> dict:
        '''
        Tokenize the context passages and returns the tokens as list of ids.
        Params:
            corpus: dict - corpus dict consisting title and text for the passages
        Returns:
            dict of outputs including tokenized ids: dict
        '''
        ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
            self.generative_model_config.dpr_pretrained_model
        )
        
        outputs = ctx_tokenizer(
            corpus['title'],
            corpus['text'],
            truncation=True,
            padding="longest",
            return_tensors='pt'
        )
        
        return outputs
        
    def __generate_embeddings(self, token_ids: dict) -> list:
        '''
        generates embeddings for the defined tokens using DPR context encoder 
        Params:
            token_ids: tensor - the output for the tokens from DPR tokenizer
        Returns:
            embeddings in batch: list
        '''
        ctx_encoder = DPRContextEncoder.from_pretrained(self.generative_model_config.dpr_pretrained_model)
        if self.device is not None:
            ctx_encoder = ctx_encoder.to(device=self.device)
            
        torch.set_grad_enabled(False)
        
        batch_size = self.generative_model_config.batch_size
        num_passages = token_ids.size()[0]
        
        embeds_batch = []
        for i in range(0, num_passages, batch_size):
            batch_ids = token_ids[i:i+batch_size, :]
            if self.device is not None:
                batch_ids = batch_ids.to(device=self.device)
            
            outputs = ctx_encoder(batch_ids, return_dict=True)
            
            embeddings = outputs['pooler_output']
            embeddings = embeddings.detach().cpu().numpy()
            
            embeds_batch.append(embeddings)
        
        return embeds_batch
    
    def __generate_indexed_dataset(self, dataset: Dataset) -> Dataset:
        '''
        adds the faiss HNSWFlat indexing to the dataset
        Params:
            dataset: Dataset - original dataset with embeddings
        Returns:
            indexed dataset: Dataset
        '''
        index = faiss.IndexHNSWFlat(
            self.generative_model_config.index_dim,
            self.generative_model_config.index_m,
            faiss.METRIC_INNER_PRODUCT
        )
        
        dataset.add_faiss_index(
            column='embeddings',
            index_name='embeddings',
            custom_index=index,
            faiss_verbose=True 
        )
        
        return dataset 
        