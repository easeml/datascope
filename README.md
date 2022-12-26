# Ease.ml/Datascope: Guiding your Data-centric Data Iterations, over End-to-end ML pipelines

[![PyPI version](https://badge.fury.io/py/datascope.svg)](https://badge.fury.io/py/datascope)

Developing ML applications are data-centric --- often *the quality of your model
is a reflection of the quality of your underlying data*. In the era of data-
centric AI, the fundamental question becomes

  > _Which training data example is most important to improve the accuracy/fairness of my ML model?_

Once you know these "importances", we can use it to support a range of applications ---
clean your data and fix your data bugs, data acquisition, data summarization, etc. 
(e.g., [https://arxiv.org/pdf/1911.07128.pdf](https://arxiv.org/pdf/1911.07128.pdf)). 

DataScope is a tool for inspecting ML pipelines by measuring how important each 
training data point is. The most prominent feature of DataScope is that it 
supports not only a single ML model, but also any `sklearn` Pipeline --- it is 
also super fast, up to four orders of magnitude faster than previous approaches.
The secret sauce of DataScope is a collection of new results on computing
the Shapley value of a specific family of ML models (K-nearest neighbor classifiers)
in PTIME, over relational data provenances. If you want to learn more about how DataScope works, 
the main reference is [https://arxiv.org/abs/2204.11131](https://arxiv.org/abs/2204.11131), and a series of our previous studies on
KNN Shapley proxies can be found at [https://ease.ml/datascope](https://ease.ml/datascope).

In just seconds, you will be able to get the importance score for each of your 
training examples, and get your data-centric cleaning/debugging iterations
started!

DataScope is part of the Ease.ML data-centric ML DevOps eco-system: [https://Ease.ML](https://Ease.ML)

## References

```
@misc{https://doi.org/10.48550/arxiv.2204.11131,
  url = {https://arxiv.org/abs/2204.11131},
  author = {Karlaš, Bojan and Dao, David and Interlandi, Matteo and Li, Bo and Schelter, Sebastian and Wu, Wentao and Zhang, Ce},
  title = {Data Debugging with Shapley Importance over End-to-End Machine Learning Pipelines},
  publisher = {arXiv}, year = {2022},
}
```

## Quick Start

Install by running:

```bash
pip install datascope
```

We can compute the Shapley importance scores for some scikit-learn pipeline `pipeline` using a training dataset `(X_train, y_train)` and a validation dataset `(X_val, y_val)` as such:

```python
from datascope.importance.common import SklearnModelAccuracy
from datascope.importance.shapley import ShapleyImportance

utility = SklearnModelAccuracy(pipeline)
importance = ShapleyImportance(method="neighbor", utility=utility)
importances = importance.fit(X_train, y_train).score(X_val, y_val)
```

The variable `importances` contains Shapley values of all data examples in `(X_train, y_train)` computed using the nearest neighbor method (i.e. `"neighbor"`).

For a more complete example workflow, see the [demo Colab notebook](https://colab.research.google.com/drive/1faCvkKLFA7m4kj8GzxBNBMMq0nXi70H3?usp=sharing).

## Why datascope?

Shapley values help you find faulty data examples ***much faster*** than if you were going about it randomly. For example, let's say you are given a dataset with 50% of labels corrupted, and you want to repair them one by one. Which one should you select first?

![Example data repair workflow using datascope](/dev/assets/uci-stdscaler-pipeline-experiment.png)

In the above figure, we run different methods for prioritizing data examples that should get repaired (random selection, various methods that use the Shapley importance). After each repair, we measure the accuracy achieved on an XGBoost model. We can see in the left figure that each importance-based method is better than random. Furthermore, for the KNN method (i.e. the `"neighbor"` method), we are able to achieve peak performance after repairing only 50% of labels.

> ease.ml/datascope speeds up data debugging by allowing you to focus on the most important data examples first

If we look at speed (right figure), we measure three different methods (the `"neighbor"` method and the `"montecarlo"` method for 10 iterations and 100 iterations). We can see that our KNN-based importance computation method is orders of magnitude faster than the state-of-the-art Monte-Carlo method.

> The "neighbor" method in ease.ml/datascope can compute importances in seconds for datasets of several thousand examples
