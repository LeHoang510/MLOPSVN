from typing import Any, Dict, List, Optional

import bentoml
import feast
import mlflow
import numpy as np
import pandas as pd
import requests
from bentoml.io import JSON
from mlflow.models.signature import ModelSignature
from pydantic import BaseModel

from utils import *

Log(AppConst.BENTOML_SERVICE)
AppPath()
pd.set_option("display.max_columns", None)
config = Config()
Log().log.info(f"config: {config.__dict__}")

def save_model() -> List[bentoml.Model]:
    Log().log.info("start save_model")
    # Read from .env file registered_model_version.json, get model names and URIs

    registered_model_files = [
        AppPath.ROOT / config.registered_model_1_file,
        AppPath.ROOT / config.registered_model_2_file
    ]

    models = []
    for registered_model_file in registered_model_files:
        Log().log.info(f"registered_model_file: {registered_model_file}")
        registered_model_dict = load_json(registered_model_file)
        Log().log.info(f"registered_model_dict: {registered_model_dict}")

        run_id = registered_model_dict["_run_id"]
        model_name = registered_model_dict["_name"]
        model_version = registered_model_dict["_version"]
        model_uri = registered_model_dict["_source"]

        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow_model = mlflow.pyfunc.load_model(model_uri=model_uri)
        Log().log.info(mlflow_model.__dict__)
        model = mlflow_model._model_impl
        model_signature: ModelSignature = mlflow_model.metadata.signature

        # Construct feature list
        feature_list = []
        for name in model_signature.inputs.input_names():
            feature_list.append(name)

        # Save model using BentoML
        bentoml_model = bentoml.sklearn.save_model(
            model_name,
            model,
            # Model signatures for runner inference
            signatures={
                "predict": {
                    "batchable": False,
                },
            },
            labels={
                "owner": "mlopsvn",
            },
            metadata={
                "mlflow_run_id": run_id,
                "mlflow_model_name": model_name,
                "mlflow_model_version": model_version,
            },
            custom_objects={
                "feature_list": feature_list,
            },
        )
        Log().log.info(bentoml_model.__dict__)
        models.append(bentoml_model)

    return models


bentoml_models = save_model()

feature_lists = []
bentoml_runners = []
for bentoml_model in bentoml_models:
    feature_list = bentoml_model.custom_objects["feature_list"]
    feature_lists.append(feature_list)
    bentoml_runner = bentoml.sklearn.get(bentoml_model.tag).to_runner()
    bentoml_runners.append(bentoml_runner)

svc1 = bentoml.Service(bentoml_models[0].tag.name, runners=[bentoml_runners[0]])
svc2 = bentoml.Service(bentoml_models[1].tag.name, runners=[bentoml_runners[1]])
fs = feast.FeatureStore(repo_path=AppPath.FEATURE_REPO)

def predict1(request: np.ndarray) -> np.ndarray:
    Log().log.info(f"start predict of model 1")
    result = bentoml_runners[0].predict.run(request)
    Log().log.info(f"result of model 1: {result}")
    return result

def predict2(request: np.ndarray) -> np.ndarray:
    Log().log.info(f"start predict of model 2")
    result = bentoml_runners[1].predict.run(request)
    Log().log.info(f"result of model 2: {result}")
    return result

class InferenceRequest(BaseModel):
    request_id: str
    driver_ids: List[int]


class InferenceResponse(BaseModel):
    prediction: Optional[float]
    error: Optional[str]

@svc1.api(
    input=JSON(pydantic_model=InferenceRequest),
    output=JSON(pydantic_model=InferenceResponse),
    route="/phase-1/prob-1/predict",
)
def inference_model1(request: InferenceRequest, ctx: bentoml.Context) -> Dict[str, Any]:
    """
    Example request: {"request_id": "uuid-1", "driver_ids":[1001,1002,1003,1004,1005]}
    """
    Log().log.info(f"start inference for Model 1")
    response = InferenceResponse()
    try:
        Log().log.info(f"request: {request}")
        driver_ids = request.driver_ids

        online_features = fs.get_online_features(
            entity_rows=[{"driver_id": driver_id} for driver_id in driver_ids],
            features=[f"driver_stats:{name}" for name in feature_lists[0]],
        )
        df = pd.DataFrame.from_dict(online_features.to_dict())
        Log().log.info(f"online features: {df}")

        input_features = df.drop(["driver_id"], axis=1)
        input_features = input_features[feature_lists[0]]
        Log().log.info(f"input_features: {input_features}")

        result = predict1(input_features)
        df["prediction"] = result
        best_idx = df["prediction"].argmax()
        best_driver_id = df["driver_id"].iloc[best_idx]
        Log().log.info(f"best_driver_id: {best_driver_id}")
        Log().log.info(f"df: {df}")

        response.prediction = best_driver_id
        ctx.response.status_code = 200

        # monitor
        monitor_df = df.iloc[[best_idx]]
        monitor_df = monitor_df.assign(request_id=[request.request_id])
        monitor_df = monitor_df.assign(best_driver_id=[best_driver_id])
        Log().log.info(f"monitor_df: {monitor_df}")
        monitor_request(monitor_df)

    except Exception as e:
        Log().log.error(f"error: {e}")
        response.error = str(e)
        ctx.response.status_code = 500

    Log().log.info(f"response: {response}")
    return response

@svc2.api(
    input=JSON(pydantic_model=InferenceRequest),
    output=JSON(pydantic_model=InferenceResponse),
    route="/phase-1/prob-2/predict",
)
def inference_model2(request: InferenceRequest, ctx: bentoml.Context) -> Dict[str, Any]:
    """
    Example request: {"request_id": "uuid-1", "driver_ids":[1001,1002,1003,1004,1005]}
    """
    Log().log.info(f"start inference for Model 1")
    response = InferenceResponse()
    try:
        Log().log.info(f"request: {request}")
        driver_ids = request.driver_ids

        online_features = fs.get_online_features(
            entity_rows=[{"driver_id": driver_id} for driver_id in driver_ids],
            features=[f"driver_stats:{name}" for name in feature_lists[0]],
        )
        df = pd.DataFrame.from_dict(online_features.to_dict())
        Log().log.info(f"online features: {df}")

        input_features = df.drop(["driver_id"], axis=1)
        input_features = input_features[feature_lists[0]]
        Log().log.info(f"input_features: {input_features}")

        result = predict2(input_features)
        df["prediction"] = result
        best_idx = df["prediction"].argmax()
        best_driver_id = df["driver_id"].iloc[best_idx]
        Log().log.info(f"best_driver_id: {best_driver_id}")
        Log().log.info(f"df: {df}")

        response.prediction = best_driver_id
        ctx.response.status_code = 200

        # monitor
        monitor_df = df.iloc[[best_idx]]
        monitor_df = monitor_df.assign(request_id=[request.request_id])
        monitor_df = monitor_df.assign(best_driver_id=[best_driver_id])
        Log().log.info(f"monitor_df: {monitor_df}")
        monitor_request(monitor_df)

    except Exception as e:
        Log().log.error(f"error: {e}")
        response.error = str(e)
        ctx.response.status_code = 500

    Log().log.info(f"response: {response}")
    return response


def monitor_request(df: pd.DataFrame):
    Log().log.info("start monitor_request")
    try:
        data = json.dumps(df.to_dict(), cls=NumpyEncoder)

        Log().log.info(f"sending {data}")
        response = requests.post(
            config.monitoring_service_api,
            data=data,
            headers={"content-type": "application/json"},
        )

        if response.status_code == 200:
            Log().log.info(f"Success")
        else:
            Log().log.info(
                f"Got an error code {response.status_code} for the data chunk. Reason: {response.reason}, error text: {response.text}"
            )

    except requests.exceptions.ConnectionError as error:
        Log().log.error(
            f"Cannot reach monitoring service, error: {error}, data: {data}"
        )

    except Exception as error:
        Log().log.error(f"Error: {error}")