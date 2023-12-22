"""
This module defines custom metric functions that are invoked during the 'train' and 'evaluate'
steps to provide model performance insights. Custom metric functions defined in this module are
referenced in the ``metrics`` section of ``recipe.yaml``, for example:

.. code-block:: yaml
    :caption: Example custom metrics definition in ``recipe.yaml``

    metrics:
      custom:
        - name: weighted_mean_squared_error
          function: weighted_mean_squared_error
          greater_is_better: False
"""
from typing import Dict

from pandas import DataFrame
from sklearn.metrics import r2_score


def get_custom_metrics(
    eval_df: DataFrame,
    builtin_metrics: Dict[str, float],  # pylint: disable=unused-argument
) -> float:
    """
    FIXME::OPTIONAL: provide function doc string.
    :param eval_df: A Pandas DataFrame containing the following columns:
                    - ``"prediction"``: Predictions produced by submitting input data to the model.
                    - ``"target"``: Ground truth values corresponding to the input data.
    :param builtin_metrics: A dictionary containing the built-in metrics that are calculated
                            automatically during model evaluation. The keys are the names of the
                            metrics and the values are the scalar values of the metrics. For more
                            information, see
                            https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate.
    :return: A numeric scalar value.
    """
    print(eval_df.columns)
    # evaluator_config={'col_mapping': {"y_true": "amount_in_usd", "y_pred": "predictions"}}
    return r2_score( eval_df["target"], eval_df["prediction"])
