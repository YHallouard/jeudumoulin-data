from cli.utils import yaml_arg

from workflows._flows import train_alphazero_flow, train_dqn_flow

IMAGE = "ghcr.io/yhallouard/jeudumoulin/jeudumoulin-worker:latest"
MLFLOW_TRACKING_URI = "http://mlflow.mlops.svc.cluster.local:5000"

if __name__ == "__main__":
    alphazero_config = yaml_arg("config/train_alphazero_light.yaml")["config"]

    train_alphazero_flow.deploy(
        name="train-alphazero-k8s",
        work_pool_name="kubernetes-homelab",
        image=IMAGE,
        build=False,
        push=False,
        job_variables={"namespace": "mlops"},
        parameters={
            "raw_config": alphazero_config,
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        },
    )

    dqn_config = yaml_arg("config/train_dqn.yaml")["config"]
    train_dqn_flow.deploy(
        name="train-dqn-k8s",
        work_pool_name="kubernetes-homelab",
        image=IMAGE,
        build=False,
        push=False,
        job_variables={"namespace": "mlops"},
        parameters={
            "raw_config": dqn_config,
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        },
    )
