import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    """
    numpy.random.seed is a function in the NumPy library that sets the seed 
    for generating random numbers. By specifying a seed value, the function
    ensures that the sequence of random numbers generated remains the same
    across multiple runs, providing deterministic behavior and allowing 
    reproducibility in random number generation.
    """

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    print("train test split is done...")
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        mlflow.set_tag("User", "Nilay Karade")
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        print("model training is completed...")
        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        
        predictions = lr.predict(train_x)
        """
        The signature represents model input and output as data frames        
        """
        signature = infer_signature(train_x, predictions)

        """
        mlflow.set_tracking_uri() connects to a tracking URI. You can also set 
        the MLFLOW_TRACKING_URI environment variable to have MLflow find a URI
        from there. In both cases, the URI can either be a HTTP/HTTPS URI for 
        a remote server, a database connection string, or a local path to log
        data to a directory. The URI defaults to mlruns.
        """
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        """
        Log a scikit-learn model as an MLflow artifact for the current run
        """
        mlflow.sklearn.log_model(lr, "model", signature=signature)