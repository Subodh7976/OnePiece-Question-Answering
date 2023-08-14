from flask import Flask, request, render_template
import os
import time

from src.pipeline.training import TrainPipelineConfig
from src.pipeline.prediction import PredictPipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import check_model_exist


class MyFlask(Flask):
    def run(self, host=None, port=None, debug=None, load_dotenv=None, **kwargs):
        if not self.debug or os.getenv('WERKZEUG_RUN_PATH') == 'true':
            with self.app_context():
                global prediction_pipeline
                if check_model_exist(TrainPipelineConfig().model_save_path) and check_model_exist(TrainPipelineConfig().generative_model_save_path):
                    prediction_pipeline = PredictPipeline()
                else:
                    prediction_pipeline = None
        super(MyFlask, self).run(host=host, port=port,
                                 debug=debug, load_dotenv=load_dotenv, **kwargs)


app = MyFlask(__name__)
prediction_pipeline = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html', trained_stand=check_model_exist(
            TrainPipelineConfig().model_save_path
        ), 
                               trained_gen=check_model_exist(
                                   TrainPipelineConfig().generative_model_save_path
                               ))
    else:
        start_time = time.time()
        gen_model_bool = bool(request.form.get('model'))
        scrape_bool = bool(request.form.get('scrape'))
        if gen_model_bool:
            if scrape_bool:
                os.system('python main.py scrape gen')
            else:
                os.system('python main.py gen')
        else:
            if scrape_bool:
                os.system('python main.py scrape')
            else:
                os.system('python main.py')

        total_time = time.time() - start_time
        global prediction_pipeline
        prediction_pipeline = PredictPipeline()

        return render_template('train.html', time_taken=total_time, 
                               completed=True)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global prediction_pipeline
    if request.method == 'GET':
        return render_template('predict.html', trained_stand=check_model_exist(
            TrainPipelineConfig().model_save_path
        ), 
                               trained_gen=check_model_exist(
                                   TrainPipelineConfig().generative_model_save_path
                               ))
    else:
        if prediction_pipeline is None:
            prediction_pipeline = PredictPipeline()
        start_time = time.time()
        model = request.form.get('model')
        query = request.form.get('query')
        generative = False
        if model == 'generative':
            generative = True
            result = prediction_pipeline.predict_generative(query=query)
        else:
            result = prediction_pipeline.predict(query=query)

        total_time = time.time() - start_time

        return render_template('predict.html', result=result,
                               completed=True,
                               time_taken=total_time,
                               trained_stand=check_model_exist(
                                   TrainPipelineConfig().model_save_path
                               ), 
                               trained_gen=check_model_exist(
                                   TrainPipelineConfig().generative_model_save_path
                               ), generative=generative)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
