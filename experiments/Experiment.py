import numpy as np

from datascope.algorithms import KNN_Shapley, TMC_Shapley
from datascope.inspection.utils import process_pipe_condknn, process_pipe_condpipe, process_pipe_knn

from loader import FashionMnist, UCI, TwentyNews

from apps import Label, Poisoning, Feature, Fairness
from plotter import (
    LabelPlotter,
    LabelCleaningPlotter,
    PoisoningPlotter,
    PoisoningCleaningPlotter,
    FeaturePlotter,
    FeatureCleaningPlotter,
    FairnessPlotter,
    RuntimePlotter,
)

from sklearn.metrics import accuracy_score
from fairlearn.metrics import true_positive_rate, false_positive_rate

from functools import partial

from matplotlib.pyplot import figure
from datetime import datetime
import seaborn as sns

import time
import os

# sns.set_theme()


class Experiment:
    def __init__(self, name, pipeline, dataset_name="FashionMNIST", save_path="."):
        self.name = name
        self.pipeline = pipeline
        self.custom_save_path = save_path
        self.base_path = ""
        self.dataset_name = dataset_name  # UCI, FashionMNIST, Text
        self.dataset_size = 1100

        if self.pipeline is None:
            raise ValueError("need a pipeline to run")

        if self.name is None:
            raise ValueError("need name for experiment")

    def run(
        self,
        iterations=1000,
        run_label=False,
        run_poisoning=False,
        run_feature=False,
        run_fairness=False,
        run_augmentation=False,
        run_performance=False,
        ray=False,
        truncated=True,
        forksets=None,
        run_forks=0,
    ):
        name = self.name
        self.run_forks = run_forks  # flag (number of forks) to run fork experiments
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        self.base_path = f"{self.custom_save_path}/results/{self.name}/i-{iterations}-time-{dt_string}"

        # flatten image datasets for sklearn
        flatten = False
        if self.dataset_name == "FashionMNIST":
            if self.pipeline != "pipe-7":
                flatten = True

        def create_dirs(path):
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                except OSError:
                    print("Creation of the directory %s failed" % path)
                else:
                    print("Successfully created the directory %s" % path)

        print(f"Initiate experiment {name}-i-{iterations}-time-{dt_string}")

        if run_label:
            print("[DataScope] => Running label noise experiment")
            create_dirs(f"{self.custom_save_path}/results/{name}/i-{iterations}-time-{dt_string}/label/")
            print(flatten)
            self.run_label_experiment(iterations, dt_string, ray, truncated, forksets=forksets, flatten=flatten)
        if run_poisoning:
            print("[DataScope] => Running poisoning experiment")
            create_dirs(f"{self.custom_save_path}/results/{name}/i-{iterations}-time-{dt_string}/poisoning/")
            self.run_poisoning_experiment(iterations, dt_string, ray, truncated, forksets=forksets, flatten=flatten)
        if run_feature:
            print("[DataScope] => Running feature noise experiment")
            create_dirs(f"{self.custom_save_path}/results/{name}/i-{iterations}-time-{dt_string}/feature/")
            self.run_feature_experiment(iterations, dt_string, ray, truncated, forksets=forksets, flatten=flatten)
        if run_fairness:
            print("[DataScope] => Running fairness experiment")
            create_dirs(f"{self.custom_save_path}/results/{name}/i-{iterations}-time-{dt_string}/fairness/")
            self.run_fairness_experiment(iterations, dt_string, ray, truncated, forksets=forksets, flatten=flatten)
        if run_augmentation:
            print("[DataScope] => Running augmentation experiment")
            create_dirs(f"{self.custom_save_path}/results/{name}/i-{iterations}-time-{dt_string}/augmentation/")
            self.run_augmentation_experiment(iterations, dt_string, ray, truncated, forksets=forksets)
        if run_performance:
            print("[DataScope] => Running label noise experiment")
            create_dirs(f"{self.custom_save_path}/results/{name}/i-{iterations}-time-{dt_string}/label/")
            print(flatten)
            self.run_performance_experiment(
                iterations, dt_string, ray, truncated=True, forksets=forksets, flatten=flatten
            )

        print("[DataScope] => All experiments done!")

    def run_label_experiment(self, iterations, dt_string, ray, truncated, forksets=None, flatten=True):
        """
        Run label noise experiment and plots evaluation
        """
        name = self.name + "_label"
        if not truncated:
            name = name + "_mc"
        num = self.dataset_size
        # loader = MNIST(num_train=num, one_hot=False, shuffle=True, by_label=True)
        if self.dataset_name == "UCI":
            loader = UCI(num_train=num)
        elif self.dataset_name == "FashionMNIST":
            loader = FashionMnist(num_train=num, flatten=flatten)
        elif self.dataset_name == "20NewsGroups":
            loader = TwentyNews(num_train=num)

        X_train, y_train, X_test, y_test = loader.prepare_data()

        measure_KNN = KNN_Shapley(K=1)
        measure_TMC = TMC_Shapley(metric=accuracy_score, iterations=iterations, ray=ray, truncated=truncated)

        app_label = Label(X_train, y_train, X_test, y_test, flatten=flatten)

        if self.run_forks > 1:
            print(f"[DataScope] => Generating {self.run_forks} interesting forks ...")
            forksets = app_label.get_interesting_forks(self.run_forks)

        # Processors looks for the 'model' element and split the pipelines into seperate parts
        transform = None
        pipeline = self.pipeline
        transform_condknn, pipeline_condknn = process_pipe_condknn(pipeline)
        transform_condpipe, pipeline_condpipe = process_pipe_condpipe(pipeline)
        transform_knn, pipeline_knn = process_pipe_knn(pipeline)

        start = time.perf_counter()
        res_label_condknn = app_label.run(
            measure_KNN,
            model_family="custom",
            transform=transform_condknn,
            pipeline=pipeline_condknn,
            forksets=forksets,
        )
        end = time.perf_counter()
        time_condknn = end - start
        print("condknn time: ", time_condknn)

        # start = time.perf_counter()
        # res_label_condknn_recomp_v1 = app_label.run(measure_KNN, model_family='custom', transform=transform_condknn, pipeline=pipeline_condknn, forksets=forksets, fixed_y=app_label.unflipped_y, recompute_v1=True)
        # end = time.perf_counter()
        # time_condknn_recomp_v1 = end - start
        # print("recompute v1 condknn time: ", time_condknn_recomp_v1)

        # start = time.perf_counter()
        # res_label_condknn_recomp = app_label.run(measure_KNN, model_family='custom', transform=transform_condknn, pipeline=pipeline_condknn, forksets=forksets, fixed_y=app_label.unflipped_y, recompute_v1=False)
        # end = time.perf_counter()
        # time_condknn_recomp = end - start
        # print("recompute v2 condknn time: ", time_condknn_recomp)

        # start = time.perf_counter()
        # res_label_condpipe = app_label.run(
        #     measure_TMC,
        #     model_family="custom",
        #     transform=transform_condpipe,
        #     pipeline=pipeline_condpipe,
        #     forksets=forksets,
        # )
        # end = time.perf_counter()
        # time_condpipe = end - start
        # print("condpipe time: ", time_condpipe)

        # start = time.perf_counter()
        # res_label_pipe = app_label.run(
        #     measure_TMC, model_family="custom", transform=transform, pipeline=pipeline, forksets=forksets
        # )
        # end = time.perf_counter()
        # time_pipe = end - start

        # start = time.perf_counter()
        # res_label_knn = app_label.run(
        #     measure_TMC, model_family="custom", transform=transform_knn, pipeline=pipeline_knn, forksets=forksets
        # )
        # end = time.perf_counter()
        # time_knn = end - start

        # datetime object containing current date and time
        # np.savez_compressed(f'{self.base_path}/label/shapley/{name}-i-{iterations}-time-{dt_string}-shapley',condknn=res_label_condknn, condpipe=res_label_condpipe, knn=res_label_knn, pipe=res_label_pipe)
        # np.savez_compressed(f'{self.base_path}/label/data/{name}-i-{iterations}-time-{dt_string}-data', X=X_train, y=y_train, X_test=X_test, y_test=y_test, flip_indices=app_label.flip_indices)
        # np.savez_compressed(f'{self.base_path}/label/time/{name}-i-{iterations}-time-{dt_string}-time', time_condknn=time_condknn, time_condpipe=time_condpipe, time_knn=time_knn, time_pipe=time_pipe)

        # # save figures
        # figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')
        # RuntimePlotter(
        #             ('KNN-Shapley (cond)', time_condknn),
        #             ('TMC-Shapley (cond)', time_condpipe),
        #             ('KNN-Shapley', time_knn),
        #             ('TMC-Shapley', time_pipe)).plot(save_path=f'{self.base_path}/label/LabelRuntime')

        LabelPlotter(
            app_label,
            ("Shapley", res_label_condknn),
            # ("KNN-Shapley recompute v1 (cond)", res_label_condknn_recomp_v1),
            # ("KNN-Shapley recompute (cond)", res_label_condknn_recomp),
            # ("TMC-Shapley (cond)", res_label_condpipe),
            # ("KNN-Shapley", res_label_knn),
            # ("TMC-Shapley", res_label_pipe),
        ).plot(save_path=f"{self.base_path}/label/Label")

        LabelCleaningPlotter(
            app_label,
            ("Shapley", res_label_condknn),
            # ("KNN-Shapley recompute v1 (cond)", res_label_condknn_recomp_v1),
            # ("KNN-Shapley recompute (cond)", res_label_condknn_recomp),
            # ("TMC-Shapley (cond)", res_label_condpipe),
            # ("KNN-Shapley", res_label_knn),
            # ("TMC-Shapley", res_label_pipe),
        ).plot(ray=ray, model_family="custom", pipeline=pipeline, save_path=f"{self.base_path}/label/LabelCleaning")

    def run_poisoning_experiment(self, iterations, dt_string, ray, truncated, forksets=None, flatten=True):
        """
        Run poisoning experiment and plots evaluation
        """
        use_type = ""
        name = self.name + "_poisoning"
        if not truncated:
            name = name + "_mc"
        num = self.dataset_size
        if self.dataset_name == "UCI":
            loader = UCI(num_train=num)
            use_type = "tabular"
        elif self.dataset_name == "FashionMNIST":
            loader = FashionMnist(num_train=num, flatten=flatten)
            use_type = "image"
        elif self.dataset_name == "20NewsGroups":
            loader = TwentyNews(num_train=num)
            use_type = "text"

        X_train, y_train, X_test, y_test = loader.prepare_data()

        measure_KNN = KNN_Shapley(K=1)
        measure_TMC = TMC_Shapley(metric=accuracy_score, iterations=iterations, ray=ray, truncated=truncated)

        app_poisoning = Poisoning(X_train, y_train, X_test, y_test, use_type=use_type)

        if self.run_forks > 1:
            print(f"[DataScope] => Generating {self.run_forks} interesting forks ...")
            forksets = app_poisoning.get_interesting_forks(self.run_forks)

        # Processors looks for the 'model' element and split the pipelines into seperate parts
        transform = None
        pipeline = self.pipeline
        transform_condknn, pipeline_condknn = process_pipe_condknn(pipeline)
        transform_condpipe, pipeline_condpipe = process_pipe_condpipe(pipeline)
        transform_knn, pipeline_knn = process_pipe_knn(pipeline)

        start = time.time()
        res_poisoning_condknn = app_poisoning.run(
            measure_KNN,
            model_family="custom",
            transform=transform_condknn,
            pipeline=pipeline_condknn,
            forksets=forksets,
        )
        end = time.time()
        time_condknn = end - start

        start = time.time()
        res_poisoning_condknn_recomp_v1 = app_poisoning.run(
            measure_KNN,
            model_family="custom",
            transform=transform_condknn,
            pipeline=pipeline_condknn,
            forksets=forksets,
            fixed_y=app_poisoning.unflipped_y,
            recompute_v1=True,
        )
        end = time.time()
        time_condknn_recomp_v1 = end - start

        start = time.time()
        res_poisoning_condknn_recomp = app_poisoning.run(
            measure_KNN,
            model_family="custom",
            transform=transform_condknn,
            pipeline=pipeline_condknn,
            forksets=forksets,
            fixed_y=app_poisoning.unflipped_y,
            recompute_v1=False,
        )
        end = time.time()
        time_condknn_recomp = end - start

        start = time.time()
        res_poisoning_condpipe = app_poisoning.run(
            measure_TMC,
            model_family="custom",
            transform=transform_condpipe,
            pipeline=pipeline_condpipe,
            forksets=forksets,
        )
        end = time.time()
        time_condpipe = end - start

        start = time.time()
        res_poisoning_knn = app_poisoning.run(
            measure_TMC, model_family="custom", transform=transform_knn, pipeline=pipeline_knn, forksets=forksets
        )
        end = time.time()
        time_knn = end - start

        start = time.time()
        res_poisoning_pipe = app_poisoning.run(
            measure_TMC, model_family="custom", transform=transform, pipeline=pipeline, forksets=forksets
        )
        end = time.time()
        time_pipe = end - start

        # datetime object containing current date and time
        # np.savez_compressed(f'{self.base_path}/poisoning/shapley/{name}-i-{iterations}-time-{dt_string}-shapley',condknn=res_poisoning_condknn, condpipe=res_poisoning_condpipe, knn=res_poisoning_knn, pipe=res_poisoning_pipe)
        # np.savez_compressed(f'{self.base_path}/poisoning/data/{name}-i-{iterations}-time-{dt_string}-data', X=X_train, y=y_train, X_test=X_test, y_test=y_test, poison_indices=app_poisoning.poison_indices)
        # np.savez_compressed(f'{self.base_path}/poisoning/time/{name}-i-{iterations}-time-{dt_string}-time', time_condknn=time_condknn, time_condpipe=time_condpipe, time_knn=time_knn, time_pipe=time_pipe)

        # save figures
        figure(num=None, figsize=(6, 4), dpi=80, facecolor="w", edgecolor="k")
        # RuntimePlotter(
        #             ('KNN-Shapley (cond)', time_condknn),
        #             ('TMC-Shapley (cond)', time_condpipe),
        #             ('KNN-Shapley', time_knn),
        #             ('TMC-Shapley', time_pipe)).plot(save_path=f'{self.base_path}/poisoning/PoisoningRuntime')

        PoisoningPlotter(
            app_poisoning,
            ("KNN-Shapley (cond)", res_poisoning_condknn),
            ("KNN-Shapley recompute v1 (cond)", res_poisoning_condknn_recomp_v1),
            ("KNN-Shapley recompute (cond)", res_poisoning_condknn_recomp),
            ("TMC-Shapley (cond)", res_poisoning_condpipe),
            ("KNN-Shapley", res_poisoning_knn),
            ("TMC-Shapley", res_poisoning_pipe),
        ).plot(save_path=f"{self.base_path}/poisoning/Poisoning")

        PoisoningCleaningPlotter(
            app_poisoning,
            ("KNN-Shapley (cond)", res_poisoning_condknn),
            ("KNN-Shapley recompute v1 (cond)", res_poisoning_condknn_recomp_v1),
            ("KNN-Shapley recompute (cond)", res_poisoning_condknn_recomp),
            ("TMC-Shapley (cond)", res_poisoning_condpipe),
            ("KNN-Shapley", res_poisoning_knn),
            ("TMC-Shapley", res_poisoning_pipe),
        ).plot(
            model_family="custom", pipeline=pipeline, ray=ray, save_path=f"{self.base_path}/poisoning/PoisoningCleaning"
        )

    def run_fairness_experiment(self, iterations, dt_string, ray, truncated, forksets=None, flatten=True):

        name = self.name + "_fairness"
        if not truncated:
            name = name + "_mc"

        num = 1500
        sensitive_feature = 9
        if self.dataset_name == "UCI":
            loader = UCI(num_train=num)
        elif self.dataset_name == "FashionMNIST":
            loader = FashionMnist(num_train=num, flatten=flatten)
        elif self.dataset_name == "20NewsGroups":
            loader = TwentyNews(num_train=num)
            use_text = True
        X_train, y_train, X_test, y_test = loader.prepare_preselected_unfair()

        # X_train, y_train = loader.create_unfair_train_data(X_train, y_train, 10)
        # X_test, y_test = loader.create_fair_test_data(X_test, y_test)

        # from fairlearn.metrics import demographic_parity_ratio
        a_indices = np.where(X_test[:, 9] == 0)[0]  # woman
        b_indices = np.where(X_test[:, 9] == 1)[0]  # man

        if self.run_forks > 1:

            number_of_forksets = 15
            forksets = np.zeros(num, dtype=int)
            size_of_sets = num // number_of_forksets
            cnt_pos = 0
            cnt_neg = 0
            fork_id = 0

            women_indices = np.where(X_train[:, 9] == 0)[0]  # woman
            men_indices = np.where(X_train[:, 9] == 1)[0]  # man

            for i in range(15):

                if i < 9:
                    num_of_neg = int(np.ceil(size_of_sets / 2))
                    num_of_pos = int(np.floor(size_of_sets / 2))
                else:
                    num_of_neg = 93
                    num_of_pos = 7
                assert (num_of_neg + num_of_pos) == size_of_sets

                forksets[women_indices[cnt_pos : (cnt_pos + num_of_pos)]] = fork_id
                forksets[men_indices[cnt_neg : (cnt_neg + num_of_neg)]] = fork_id
                # print(cnt_pos, len(self.flip_indices[cnt_pos:(cnt_pos+num_of_pos)]))
                # print(cnt_neg, len(notflip_indices[(cnt_neg):(cnt_neg+num_of_neg)]))
                # keep track of counter
                cnt_pos += num_of_pos
                cnt_neg += num_of_neg
                fork_id += 1

            # print("left_over", women_indices.shape[0] - cnt_pos)
            # print("left_over", men_indices.shape[0] - cnt_neg)

            np.where(forksets == 14)

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
        # np.savez_compressed(f'{self.base_path}/fairness/{name}-i-{iterations}-time-{dt_string}-data', X=X_train, y=y_train, X_test=X_test, y_test=y_test)

        transform_condknn, pipeline_condknn = process_pipe_condknn(pipeline)
        transform_condpipe, pipeline_condpipe = process_pipe_condpipe(pipeline)
        transform_knn, pipeline_knn = process_pipe_knn(pipeline)

        start = time.time()
        res_fairness_condpipe = app_fairness.run(
            measure_TMC_fair,
            model_family="custom",
            transform=transform_condpipe,
            pipeline=pipeline_condpipe,
            forksets=forksets,
        )
        end = time.time()
        time_condpipe = end - start

        start = time.time()
        res_fairness_knn = app_fairness.run(
            measure_TMC_fair, model_family="custom", transform=transform_knn, pipeline=pipeline_knn, forksets=forksets
        )
        end = time.time()
        time_knn = end - start

        start = time.time()
        res_fairness_pipe = app_fairness.run(
            measure_TMC_fair, model_family="custom", transform=transform, pipeline=pipeline, forksets=forksets
        )
        end = time.time()
        time_pipe = end - start

        # datetime object containing current date and time
        # np.savez_compressed(f'{self.base_path}/fairness/shapley/{name}-i-{iterations}-time-{dt_string}-shapley', condpipe=res_fairness_condpipe, knn=res_fairness_knn, pipe=res_fairness_pipe)
        # np.savez_compressed(f'{self.base_path}/fairness/time/{name}-i-{iterations}-time-{dt_string}-time', time_condpipe=time_condpipe, time_knn=time_knn, time_pipe=time_pipe)

        figure(num=None, figsize=(6, 4), dpi=80, facecolor="w", edgecolor="k")

        FairnessPlotter(
            app_fairness,
            ("TMC-Shapley (cond)", res_fairness_condpipe),
            ("KNN-Shapley", res_fairness_knn),
            ("TMC-Shapley", res_fairness_pipe),
        ).plot(
            metric=eo_sex,
            metric_name="Equalized Odds Ratio",
            ray=ray,
            model_family="custom",
            pipeline=pipeline,
            save_path=f"{self.base_path}/fairness/FairnessEO",
        )

        FairnessPlotter(
            app_fairness,
            ("TMC-Shapley (cond)", res_fairness_condpipe),
            ("KNN-Shapley", res_fairness_knn),
            ("TMC-Shapley", res_fairness_pipe),
        ).plot(
            metric=dpr_sex,
            metric_name="Demographic Parity Ratio",
            ray=ray,
            model_family="custom",
            pipeline=pipeline,
            save_path=f"{self.base_path}/fairness/FairnessDPR",
        )

        FairnessPlotter(
            app_fairness,
            ("TMC-Shapley (cond)", res_fairness_condpipe),
            ("KNN-Shapley", res_fairness_knn),
            ("TMC-Shapley", res_fairness_pipe),
        ).plot(
            metric=accuracy_score,
            metric_name="Accuracy",
            ray=ray,
            model_family="custom",
            pipeline=pipeline,
            save_path=f"{self.base_path}/fairness/FairnessAccuracy",
        )

        # RuntimePlotter(
        #      ('TMC-Shapley (cond)', time_condpipe),
        #      ('KNN-Shapley', time_knn),
        #      ('TMC-Shapley', time_pipe)).plot(save_path=f'{self.base_path}/fairness/FairnessRuntime')

    def run_feature_experiment(self, iterations, dt_string, ray, truncated, forksets=None, flatten=True):
        """
        Run poisoning experiment and plots evaluation
        """
        name = self.name + "_feature"
        if not truncated:
            name = name + "_mc"
        num = self.dataset_size
        if self.dataset_name == "UCI":
            loader = UCI(num_train=num)
            use_type = "tabular"
        elif self.dataset_name == "FashionMNIST":
            loader = FashionMnist(num_train=num, flatten=flatten)
            use_type = "image"
        elif self.dataset_name == "20NewsGroups":
            loader = TwentyNews(num_train=num)
            use_type = "text"
        X_train, y_train, X_test, y_test = loader.prepare_data()

        measure_KNN = KNN_Shapley(K=1)
        measure_TMC = TMC_Shapley(metric=accuracy_score, iterations=iterations, ray=ray, truncated=truncated)

        app_feature = Feature(X_train, y_train, X_test, y_test, noisy_index=9, sigma=10, use_type=use_type)

        if self.run_forks > 1:
            print(f"[DataScope] => Generating {self.run_forks} interesting forks ...")
            forksets = app_feature.get_interesting_forks(self.run_forks)
            print(np.unique(forksets))

        # Processors looks for the 'model' element and split the pipelines into seperate parts
        transform = None
        pipeline = self.pipeline
        transform_condknn, pipeline_condknn = process_pipe_condknn(pipeline)
        transform_condpipe, pipeline_condpipe = process_pipe_condpipe(pipeline)
        transform_knn, pipeline_knn = process_pipe_knn(pipeline)

        start = time.time()
        res_feature_condknn = app_feature.run(
            measure_KNN,
            model_family="custom",
            transform=transform_condknn,
            pipeline=pipeline_condknn,
            forksets=forksets,
        )
        end = time.time()
        time_condknn = end - start

        start = time.time()
        res_feature_condpipe = app_feature.run(
            measure_TMC,
            model_family="custom",
            transform=transform_condpipe,
            pipeline=pipeline_condpipe,
            forksets=forksets,
        )
        end = time.time()
        time_condpipe = end - start

        start = time.time()
        res_feature_knn = app_feature.run(
            measure_TMC, model_family="custom", transform=transform_knn, pipeline=pipeline_knn, forksets=forksets
        )
        end = time.time()
        time_knn = end - start

        start = time.time()
        res_feature_pipe = app_feature.run(
            measure_TMC, model_family="custom", transform=transform, pipeline=pipeline, forksets=forksets
        )
        end = time.time()
        time_pipe = end - start

        # datetime object containing current date and time
        # np.savez_compressed(f'{self.base_path}/feature/shapley/{name}-i-{iterations}-time-{dt_string}-shapley',condknn=res_feature_condknn, condpipe=res_feature_condpipe, knn=res_feature_knn, pipe=res_feature_pipe)
        # np.savez_compressed(f'{self.base_path}/feature/data/{name}-i-{iterations}-time-{dt_string}-data', X=X_train, y=y_train, X_test=X_test, y_test=y_test, poison_indices=app_feature.poison_indices)
        # np.savez_compressed(f'{self.base_path}/feature/time/{name}-i-{iterations}-time-{dt_string}-time', time_condknn=time_condknn, time_condpipe=time_condpipe, time_knn=time_knn, time_pipe=time_pipe)

        # save figures
        figure(num=None, figsize=(6, 4), dpi=80, facecolor="w", edgecolor="k")
        # RuntimePlotter(
        #             ('KNN-Shapley (cond)', time_condknn),
        #             ('TMC-Shapley (cond)', time_condpipe),
        #             ('KNN-Shapley', time_knn),
        #             ('TMC-Shapley', time_pipe)).plot(save_path=f'{self.base_path}/feature/featureRuntime')

        FeaturePlotter(
            app_feature,
            ("KNN-Shapley (cond)", res_feature_condknn),
            ("TMC-Shapley (cond)", res_feature_condpipe),
            ("KNN-Shapley", res_feature_knn),
            ("TMC-Shapley", res_feature_pipe),
        ).plot(save_path=f"{self.base_path}/feature/Feature")

        FeatureCleaningPlotter(
            app_feature,
            ("KNN-Shapley (cond)", res_feature_condknn),
            ("TMC-Shapley (cond)", res_feature_condpipe),
            ("KNN-Shapley", res_feature_knn),
            ("TMC-Shapley", res_feature_pipe),
        ).plot(model_family="custom", pipeline=pipeline, ray=ray, save_path=f"{self.base_path}/feature/FeatureCleaning")

    def run_augmentation_experiment(self, iterations, dt_string, ray, truncated, forksets=None, flatten=True):
        """
        Run augmentation experiment and plots evaluation
        """
        name = self.name + "_augment"
        if not truncated:
            name = name + "_mc"
        num = self.dataset_size
        loader = FashionMnist(num_train=num, flatten=flatten)
        forksets = loader.augment()
        X_train, y_train, X_test, y_test = loader.prepare_data()

        measure_KNN = KNN_Shapley(K=1)
        measure_TMC = TMC_Shapley(metric=accuracy_score, iterations=iterations, ray=ray, truncated=truncated)

        app_aug = Fairness(X_train, y_train, X_test, y_test)

        # Processors looks for the 'model' element and split the pipelines into seperate parts
        transform = None
        pipeline = self.pipeline
        transform_condknn, pipeline_condknn = process_pipe_condknn(pipeline)
        transform_condpipe, pipeline_condpipe = process_pipe_condpipe(pipeline)
        transform_knn, pipeline_knn = process_pipe_knn(pipeline)

        start = time.time()
        res_aug_condknn = app_aug.run(
            measure_KNN,
            model_family="custom",
            transform=transform_condknn,
            pipeline=pipeline_condknn,
            forksets=forksets,
        )
        end = time.time()
        time_condknn = end - start

        start = time.time()
        res_aug_condpipe = app_aug.run(
            measure_TMC,
            model_family="custom",
            transform=transform_condpipe,
            pipeline=pipeline_condpipe,
            forksets=forksets,
        )
        end = time.time()
        time_condpipe = end - start

        start = time.time()
        res_aug_knn = app_aug.run(
            measure_TMC, model_family="custom", transform=transform_knn, pipeline=pipeline_knn, forksets=forksets
        )
        end = time.time()
        time_knn = end - start

        start = time.time()
        res_aug_pipe = app_aug.run(
            measure_TMC, model_family="custom", transform=transform, pipeline=pipeline, forksets=forksets
        )
        end = time.time()
        time_pipe = end - start

        # save figures
        figure(num=None, figsize=(6, 4), dpi=80, facecolor="w", edgecolor="k")
        # RuntimePlotter(
        #             ('KNN-Shapley (cond)', time_condknn),
        #             ('TMC-Shapley (cond)', time_condpipe),
        #             ('KNN-Shapley', time_knn),
        #             ('TMC-Shapley', time_pipe)).plot(save_path=f'{self.base_path}/poisoning/PoisoningRuntime')

        FairnessPlotter(
            app_aug,
            ("KNN-Shapley (cond)", res_aug_condpipe),
            ("TMC-Shapley (cond)", res_aug_condpipe),
            ("KNN-Shapley", res_aug_knn),
            ("TMC-Shapley", res_aug_pipe),
        ).plot(
            metric=accuracy_score,
            metric_name="Accuracy",
            ray=ray,
            model_family="custom",
            pipeline=pipeline,
            save_path=f"{self.base_path}/augmentation/Accuracy",
        )
