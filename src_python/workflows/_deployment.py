from workflows._flows import train_alphazero_flow

IMAGE = "ghcr.io/yhallouard/jeudumoulin/jeudumoulin-worker:latest"

if __name__ == "__main__":
    train_alphazero_flow.deploy(
        name="train-alphazero-k8s",
        work_pool_name="kubernetes-homelab",
        image=IMAGE,
        parameters={
            "config_path": "config/train_alphazero_light.yaml",
            "mlflow_tracking_uri": "http://mlflow.mlops.svc.cluster.local:5000",
        },
    )

    # train_dqn_flow.deploy(
    #     name="train-dqn-k8s",
    #     work_pool_name="kubernetes-homelab",
    #     parameters={
    #         "config_path": "config/train_dqn.yaml",
    #         "mlflow_tracking_uri": "http://mlflow.mlops.svc.cluster.local:5000",
    #     },
    # )

    # train_alphazero_flow.deploy(
    #     name="train-alphazero-local",
    #     work_pool_name="local-dev",
    #     parameters={
    #         "config_path": "config/train_alphazero.yaml",
    #         "mlflow_tracking_uri": "https://mlflow.yannhallouard.com",
    #     },
    # )

    # train_dqn_flow.deploy(
    #     name="train-dqn-local",
    #     work_pool_name="local-dev",
    #     parameters={
    #         "config_path": "config/train_dqn.yaml",
    #         "mlflow_tracking_uri": "https://mlflow.yannhallouard.com",
    #     },
    # )
