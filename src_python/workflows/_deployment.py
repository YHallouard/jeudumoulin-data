from cli.train._train_alphazero import TrainAlphazeroConfig
from cli.train._train_dqn import TrainDQNConfig
from cli.utils import yaml_arg

from workflows._flows import (
    TrainAlphazeroInputs,
    TrainDQNInputs,
    evaluate_alphazero_flow,
    train_alphazero_flow,
    train_dqn_flow,
)

IMAGE = "ghcr.io/yhallouard/jeudumoulin/jeudumoulin-worker:v1.5.0.dev6"
MLFLOW_TRACKING_URI = "http://mlflow.mlops.svc.cluster.local"

if __name__ == "__main__":
    alphazero_config = TrainAlphazeroConfig.model_validate(yaml_arg("config/train_alphazero_light.yaml")["config"])

    train_alphazero_flow.deploy(
        name="train-alphazero-k8s",
        work_pool_name="kubernetes-homelab",
        image=IMAGE,
        build=False,
        push=False,
        job_variables={"namespace": "mlops"},
        parameters={
            "inputs": TrainAlphazeroInputs(config=alphazero_config, mlflow_tracking_uri=MLFLOW_TRACKING_URI).model_dump(),
        },
    )

    evaluate_alphazero_flow.deploy(
        name="evaluate-alphazero-k8s",
        work_pool_name="kubernetes-homelab",
        image=IMAGE,
        build=False,
        push=False,
        job_variables={"namespace": "mlops"},
    )

    dqn_config = TrainDQNConfig.model_validate(yaml_arg("config/train_dqn.yaml")["config"])
    train_dqn_flow.deploy(
        name="train-dqn-k8s",
        work_pool_name="kubernetes-homelab",
        image=IMAGE,
        build=False,
        push=False,
        job_variables={"namespace": "mlops"},
        parameters={
            "inputs": TrainDQNInputs(config=dqn_config, mlflow_tracking_uri=MLFLOW_TRACKING_URI).model_dump(),
        },
    )
