"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer

def catfeature_label_ecoding(df: DataFrame):
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df

def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    #
    # return a scikit-learn-compatible transformer object.
    #
    # Identity feature transformation is applied when None is returned.
    import sklearn

    function_transformer_params = (
        {}
        if sklearn.__version__.startswith("1.0")
        else {"feature_names_out": "one-to-one"}
    )
    return Pipeline(
        steps=[
            (
                "calculate_time_and_duration_features",
                FunctionTransformer(catfeature_label_ecoding, **function_transformer_params),
            ),
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        (
                            "std_scaler",
                            StandardScaler(),
                            ['CUSTOMER_ORDER_ID', 'SALES_ORG', 'DISTRIBUTION_CHANNEL', 'DIVISION',
                            'RELEASED_CREDIT_VALUE', 'PURCHASE_ORDER_TYPE', 'COMPANY_CODE',
                            'ORDER_CREATION_DATE', 'ORDER_CREATION_TIME', 'CREDIT_CONTROL_AREA',
                            'SOLD_TO_PARTY', 'ORDER_AMOUNT', 'REQUESTED_DELIVERY_DATE',
                            'ORDER_CURRENCY', 'CREDIT_STATUS', 'CUSTOMER_NUMBER', 'uniue_cust_id'],
                        ),
                    ]
                ),
            ),
        ]
    )

