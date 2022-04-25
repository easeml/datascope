# ease.ml/datascope: Data Debugging for end-to-end ML pipelines

This is a tool for inspecting ML pipelines by measuring how important is each training data point for predicting the label of a given test data example.

## Quick Start

Install by running:

```bash
pip install datascope
```

We can compute the Shapley importance scores for some scikit-learn pipeline `pipeline` using a training dataset `(X_train, y_train)` and a valiation dataset `(X_val, y_val)` as such:

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

In the above figure we run different methods for prioritizing data examples that should get repaired (random selection, various methods that use the Shapley importance). After each repair, we measure the accuracy achieved on an XGBoost model. We can see in the left figure that each importance-based method is better than random. Furthermore, for the KNN method (i.e. the `"neighbor"` method), we are able to achieve peak performance after repairing only 50% of labels.

> ease.ml/datascope speeds up data debugging by allowing you to focus on the most important data examples first

If we look at speed (right figure), we measure three different methods (the `"neighbor"` method and the `"montecarlo"` method for 10 iterations and 100 iterations). We can see that our KNN-based importance computation method is orders of magnitude faster than the state-of-the-art MonteCarlo method.

> The "neighbor" method in ease.ml/datascope can compute importances in seconds for datasets of several thousand examples
