import numpy as np

from datascope.algorithms import KNN_Shapley, TMC_Shapley
from datascope.inspection.utils import process_pipe_condknn, process_pipe_condpipe, process_pipe_knn

from loader import FashionMnist, UCI, HateSpeech
from apps import Label, Poisoning, Fairness
from plotter import LabelPlotter, LabelCleaningPlotter, PoisoningPlotter, PoisoningCleaningPlotter, FairnessPlotter, RuntimePlotter

from sklearn.metrics import accuracy_score
from fairlearn.metrics import true_positive_rate, false_positive_rate

from functools import partial

from matplotlib.pyplot import figure
from datetime import datetime
import seaborn as sns

import time
import os

sns.set_theme()


class Experiment:

    def __init__(self, name, pipeline, dataset_name="FashionMNIST"):
        self.name = name
        self.pipeline = pipeline
        self.dataset_name = dataset_name #UCI, FashionMNIST, Text

        if self.pipeline is None:
            raise ValueError("need a pipeline to run")

        if self.name is None:
            raise ValueError("need name for experiment")

    def run(self, iterations=1000, run_label=False, run_poisoning=False, run_fairness=False, ray=False, truncated=True, flatten=True):
        name = self.name
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

        def create_dirs(path):
            if (not os.path.exists(path)):
                try:
                    os.makedirs(path)
                except OSError:
                    print ("Creation of the directory %s failed" % path)
                else:
                    print ("Successfully created the directory %s" % path)

        print(f'Initiate experiment {name}-i-{iterations}-time-{dt_string}')

        # create folder structure
        for exp in ['label', 'poisoning', 'fairness']:
            for folder in ['shapley', 'data', 'time', 'plots']:
                create_dirs(f'./results/{name}/i-{iterations}-time-{dt_string}/{exp}/{folder}')

        if run_label:
            print('Running label noise experiment')
            print(flatten)
            self.run_label_experiment(iterations, dt_string, ray, truncated, flatten=flatten)
        if run_poisoning:
            print('Running poisoning experiment')
            self.run_poisoning_experiment(iterations, dt_string, ray, truncated, flatten=flatten)
        if run_fairness:
            print('Running fairness experiment')
            self.run_fairness_experiment(iterations, dt_string, ray, truncated)

        print('done!')

    def run_label_experiment(self, iterations, dt_string, ray, truncated, flatten=True):
        '''
        Run label noise experiment and plots evaluation
        '''
        name = self.name + '_label'
        if not truncated:
            name = name + '_mc'
        num = 1000
        #loader = MNIST(num_train=num, one_hot=False, shuffle=True, by_label=True)
        if self.dataset_name == 'UCI':
            loader = UCI(num_train=num)
        elif self.dataset_name == 'FashionMNIST':
            loader = FashionMnist(num_train=num, flatten=flatten)
        
        X_train, y_train, X_test, y_test = loader.prepare_data()

        measure_KNN = KNN_Shapley(K=1)
        measure_TMC = TMC_Shapley(metric=accuracy_score, iterations=iterations, ray=ray, truncated=truncated)

        app_label = Label(X_train, y_train, X_test, y_test, flatten=flatten)

        # Processors looks for the 'model' element and split the pipelines into seperate parts
        transform = None
        pipeline = self.pipeline
        transform_condknn, pipeline_condknn = process_pipe_condknn(pipeline)
        transform_condpipe, pipeline_condpipe = process_pipe_condpipe(pipeline)
        transform_knn, pipeline_knn = process_pipe_knn(pipeline)

        start = time.perf_counter()
        res_label_condknn = app_label.run(measure_KNN, model_family='custom', transform=transform_condknn, pipeline=pipeline_condknn)
        end = time.perf_counter()
        time_condknn = end - start
        print("condknn time: ", time_condknn)

        start = time.perf_counter()
        res_label_condpipe = app_label.run(measure_TMC, model_family='custom', transform=transform_condpipe, pipeline=pipeline_condpipe)
        end = time.perf_counter()
        time_condpipe = end - start
        print("condpipe time: ", time_condpipe)

        start = time.perf_counter()
        res_label_knn = app_label.run(measure_TMC, model_family='custom', transform=transform_knn, pipeline=pipeline_knn)
        end = time.perf_counter()
        time_knn = end - start

        start = time.perf_counter()
        res_label_pipe = app_label.run(measure_TMC, model_family='custom', transform=transform, pipeline=pipeline)
        end = time.perf_counter()
        time_pipe = end - start

        # datetime object containing current date and time
        # np.savez_compressed(f'./results/{self.name}/i-{iterations}-time-{dt_string}/label/shapley/{name}-i-{iterations}-time-{dt_string}-shapley',condknn=res_label_condknn, condpipe=res_label_condpipe, knn=res_label_knn, pipe=res_label_pipe)
        # np.savez_compressed(f'./results/{self.name}/i-{iterations}-time-{dt_string}/label/data/{name}-i-{iterations}-time-{dt_string}-data', X=X_train, y=y_train, X_test=X_test, y_test=y_test, flip_indices=app_label.flip_indices)
        # np.savez_compressed(f'./results/{self.name}/i-{iterations}-time-{dt_string}/label/time/{name}-i-{iterations}-time-{dt_string}-time', time_condknn=time_condknn, time_condpipe=time_condpipe, time_knn=time_knn, time_pipe=time_pipe)

        # # save figures
        # figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')
        RuntimePlotter( 
                    ('KNN-Shapley (cond)', time_condknn), 
                    ('TMC-Shapley (cond)', time_condpipe), 
                    ('KNN-Shapley', time_knn), 
                    ('TMC-Shapley', time_pipe)).plot(save_path=f'./results/{self.name}/i-{iterations}-time-{dt_string}/label/plots/LabelRuntime-{name}-i-{iterations}-time-{dt_string}.png')

        LabelPlotter(app_label, 
                    ('KNN-Shapley (cond)', res_label_condknn), 
                    ('TMC-Shapley (cond)', res_label_condpipe), 
                    ('KNN-Shapley', res_label_knn), 
                    ('TMC-Shapley', res_label_pipe)).plot(save_path=f'./results/{self.name}/i-{iterations}-time-{dt_string}/label/plots/Label-{name}-i-{iterations}-time-{dt_string}.png')
        
        LabelCleaningPlotter(app_label, 
                     ('KNN-Shapley (cond)', res_label_condknn), 
                     ('TMC-Shapley (cond)', res_label_condpipe), 
                     ('KNN-Shapley', res_label_knn), 
                     ('TMC-Shapley', res_label_pipe)
                    ).plot(ray=ray, model_family='custom', pipeline=pipeline, save_path=f'./results/{self.name}/i-{iterations}-time-{dt_string}/label/plots/LabelCleaning-{name}-i-{iterations}-time-{dt_string}.png')

    def run_poisoning_experiment(self, iterations, dt_string, ray, truncated, flatten=True):
        '''
        Run poisoning experiment and plots evaluation
        '''
        name = self.name + '_poisoning'
        if not truncated:
            name = name + '_mc'
        num = 1000        
        loader = FashionMnist(num_train=num, flatten=flatten)
        X_train, y_train, X_test, y_test = loader.prepare_data()

        measure_KNN = KNN_Shapley(K=1)
        measure_TMC = TMC_Shapley(metric=accuracy_score, iterations=iterations, ray=ray, truncated=truncated)

        app_poisoning = Poisoning(X_train, y_train, X_test, y_test)

        # Processors looks for the 'model' element and split the pipelines into seperate parts
        transform = None
        pipeline = self.pipeline
        transform_condknn, pipeline_condknn = process_pipe_condknn(pipeline)
        transform_condpipe, pipeline_condpipe = process_pipe_condpipe(pipeline)
        transform_knn, pipeline_knn = process_pipe_knn(pipeline)

        start = time.time()
        res_poisoning_condknn = app_poisoning.run(measure_KNN, model_family='custom', transform=transform_condknn, pipeline=pipeline_condknn)
        end = time.time()
        time_condknn = end - start

        start = time.time()
        res_poisoning_condpipe = app_poisoning.run(measure_TMC, model_family='custom', transform=transform_condpipe, pipeline=pipeline_condpipe)
        end = time.time()
        time_condpipe = end - start

        start = time.time()
        res_poisoning_knn = app_poisoning.run(measure_TMC, model_family='custom', transform=transform_knn, pipeline=pipeline_knn)
        end = time.time()
        time_knn = end - start

        start = time.time()
        res_poisoning_pipe = app_poisoning.run(measure_TMC, model_family='custom', transform=transform, pipeline=pipeline)
        end = time.time()
        time_pipe = end - start

        # datetime object containing current date and time
        np.savez_compressed(f'./results/{self.name}/i-{iterations}-time-{dt_string}/poisoning/shapley/{name}-i-{iterations}-time-{dt_string}-shapley',condknn=res_poisoning_condknn, condpipe=res_poisoning_condpipe, knn=res_poisoning_knn, pipe=res_poisoning_pipe)
        np.savez_compressed(f'./results/{self.name}/i-{iterations}-time-{dt_string}/poisoning/data/{name}-i-{iterations}-time-{dt_string}-data', X=X_train, y=y_train, X_test=X_test, y_test=y_test, poison_indices=app_poisoning.poison_indices)
        np.savez_compressed(f'./results/{self.name}/i-{iterations}-time-{dt_string}/poisoning/time/{name}-i-{iterations}-time-{dt_string}-time', time_condknn=time_condknn, time_condpipe=time_condpipe, time_knn=time_knn, time_pipe=time_pipe)

        # save figures
        figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')
        RuntimePlotter( 
                    ('KNN-Shapley (cond)', time_condknn), 
                    ('TMC-Shapley (cond)', time_condpipe), 
                    ('KNN-Shapley', time_knn), 
                    ('TMC-Shapley', time_pipe)).plot(save_path=f'./results/{self.name}/i-{iterations}-time-{dt_string}/poisoning/plots/PoisoningRuntime-{name}-i-{iterations}-time-{dt_string}.png')

        PoisoningPlotter(app_poisoning, 
                    ('KNN-Shapley (cond)', res_poisoning_condknn), 
                    ('TMC-Shapley (cond)', res_poisoning_condpipe), 
                    ('KNN-Shapley', res_poisoning_knn), 
                    ('TMC-Shapley', res_poisoning_pipe)).plot(save_path=f'./results/{self.name}/i-{iterations}-time-{dt_string}/poisoning/plots/Poisoning-{name}-i-{iterations}-time-{dt_string}.png')
        
        PoisoningCleaningPlotter(app_poisoning, 
                     ('KNN-Shapley (cond)', res_poisoning_condknn), 
                     ('TMC-Shapley (cond)', res_poisoning_condpipe), 
                     ('KNN-Shapley', res_poisoning_knn), 
                     ('TMC-Shapley', res_poisoning_pipe)
                     ).plot(model_family='custom', pipeline=pipeline, save_path=f'./results/{self.name}/i-{iterations}-time-{dt_string}/poisoning/plots/PoisoningCleaning-{name}-i-{iterations}-time-{dt_string}.png')

    def run_fairness_experiment(self, iterations, dt_string, ray, truncated):

        name = self.name + '_fairness'
        if not truncated:
            name = name + '_mc'

        num = 10000
        sensitive_feature = 9
        loader = UCI(num_train=num)
        X_train, y_train, X_test, y_test = loader.prepare_preselected_unfair()

        #X_train, y_train = loader.create_unfair_train_data(X_train, y_train, 10)
        #X_test, y_test = loader.create_fair_test_data(X_test, y_test)

        #from fairlearn.metrics import demographic_parity_ratio
        a_indices = np.where(X_test[:,9] == 0)[0] #woman
        b_indices = np.where(X_test[:,9] == 1)[0] #man

        def dpr_precomputed(y, y_pred, a_indices=a_indices, b_indices=b_indices):
            count_a = y_pred[a_indices].sum()
            count_b = y_pred[b_indices].sum()
            
            # prevent nan cases
            if count_a == 0:
                return 0
            if count_b == 0:
                return 0

            if count_a > count_b:
                res = count_b / count_a
            else:
                res = count_a / count_b
            return res

        def equal_odds(y, y_pred, a_indices=a_indices, b_indices=b_indices):
            
            # True positive ratio
            tpr_a = true_positive_rate(y[a_indices], y_pred[a_indices])
            tpr_b = true_positive_rate(y[b_indices], y_pred[b_indices])

            tpr = 0
            if tpr_a > tpr_b:
                tpr = tpr_b / tpr_a
            else:
                tpr = tpr_a / tpr_b

            # False positive ratio
            fpr_a = false_positive_rate(y[a_indices], y_pred[a_indices])
            fpr_b = false_positive_rate(y[b_indices], y_pred[b_indices])

            fpr = 0
            if fpr_a > fpr_b:
                fpr = fpr_b / fpr_a
            else:
                fpr = fpr_a / fpr_b

            if tpr > fpr:
                return fpr
            else:
                return tpr

        dpr_sex = partial(dpr_precomputed, a_indices=a_indices, b_indices=b_indices)
        eo_sex = partial(equal_odds, a_indices=a_indices, b_indices=b_indices)
        measure_TMC_fair = TMC_Shapley(metric=eo_sex, iterations=iterations, ray=ray, truncated=truncated)

        app_fairness = Fairness(X_train, y_train, X_test, y_test)

        # Processors looks for the 'model' element and split the pipelines into seperate parts
        transform = None
        pipeline = self.pipeline
        pipeline.fit(X_train, y_train)
        dpr_score = dpr_precomputed(y_test, pipeline.predict(X_test))
        eo_score = equal_odds(y_test, pipeline.predict(X_test))
        print(f"Initial demographic ratio parity is {dpr_score}!")
        print(f"Initial equalized odds ratio is {eo_score}!")
        np.savez_compressed(f'./results/{self.name}/i-{iterations}-time-{dt_string}/fairness/data/{name}-i-{iterations}-time-{dt_string}-data', X=X_train, y=y_train, X_test=X_test, y_test=y_test)

        transform_condknn, pipeline_condknn = process_pipe_condknn(pipeline)
        transform_condpipe, pipeline_condpipe = process_pipe_condpipe(pipeline)
        transform_knn, pipeline_knn = process_pipe_knn(pipeline)

        start = time.time()
        res_fairness_condpipe = app_fairness.run(measure_TMC_fair, model_family='custom', transform=transform_condpipe, pipeline=pipeline_condpipe)
        end = time.time()
        time_condpipe = end - start

        start = time.time()
        res_fairness_knn = app_fairness.run(measure_TMC_fair, model_family='custom', transform=transform_knn, pipeline=pipeline_knn)
        end = time.time()
        time_knn = end - start

        start = time.time()
        res_fairness_pipe = app_fairness.run(measure_TMC_fair, model_family='custom', transform=transform, pipeline=pipeline)
        end = time.time()
        time_pipe = end - start

        # datetime object containing current date and time
        np.savez_compressed(f'./results/{self.name}/i-{iterations}-time-{dt_string}/fairness/shapley/{name}-i-{iterations}-time-{dt_string}-shapley', condpipe=res_fairness_condpipe, knn=res_fairness_knn, pipe=res_fairness_pipe)
        np.savez_compressed(f'./results/{self.name}/i-{iterations}-time-{dt_string}/fairness/time/{name}-i-{iterations}-time-{dt_string}-time', time_condpipe=time_condpipe, time_knn=time_knn, time_pipe=time_pipe)

        figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')

        FairnessPlotter(app_fairness, 
             ('TMC-Shapley (cond)', res_fairness_condpipe), 
             ('KNN-Shapley', res_fairness_knn), 
             ('TMC-Shapley', res_fairness_pipe)
                ).plot(metric=eo_sex, metric_name="Equalized Odds Ratio", model_family='custom', pipeline=pipeline, save_path=f'./results/{self.name}/i-{iterations}-time-{dt_string}/fairness/plots/FairnessEO-{name}-i-{iterations}-time-{dt_string}.png')

        FairnessPlotter(app_fairness, 
             ('TMC-Shapley (cond)', res_fairness_condpipe), 
             ('KNN-Shapley', res_fairness_knn), 
             ('TMC-Shapley', res_fairness_pipe)
                ).plot(metric=dpr_sex, metric_name="Demographic Parity Ratio", model_family='custom', pipeline=pipeline, save_path=f'./results/{self.name}/i-{iterations}-time-{dt_string}/fairness/plots/FairnessDPR-{name}-i-{iterations}-time-{dt_string}.png')

        FairnessPlotter(app_fairness, 
             ('TMC-Shapley (cond)', res_fairness_condpipe), 
             ('KNN-Shapley', res_fairness_knn), 
             ('TMC-Shapley', res_fairness_pipe)
                ).plot(metric=accuracy_score, metric_name='Accuracy', model_family='custom', pipeline=pipeline, save_path=f'./results/{self.name}/i-{iterations}-time-{dt_string}/fairness/plots/FairnessAccuracy-{name}-i-{iterations}-time-{dt_string}.png')

        RuntimePlotter( 
             ('TMC-Shapley (cond)', time_condpipe), 
             ('KNN-Shapley', time_knn), 
             ('TMC-Shapley', time_pipe)).plot(save_path=f'./results/{self.name}/i-{iterations}-time-{dt_string}/fairness/plots/FairnessRuntime-{name}-i-{iterations}-time-{dt_string}.png')