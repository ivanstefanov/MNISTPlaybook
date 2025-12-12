import kfp
import kfp.dsl as dsl
from kfp import components

from kubeflow.katib import ApiClient
from kubeflow.katib import V1beta1ExperimentSpec
from kubeflow.katib import V1beta1AlgorithmSpec
from kubeflow.katib import V1beta1ObjectiveSpec
from kubeflow.katib import V1beta1ParameterSpec
from kubeflow.katib import V1beta1FeasibleSpace
from kubeflow.katib import V1beta1TrialTemplate
from kubeflow.katib import V1beta1TrialParameterSpec

# based on https://github.com/kubeflow/katib/tree/master/examples/v1beta1/kubeflow-pipelines#multi-user-pipelines-setup
def create_katib_experiment_task(
    experiment_name: str,
    experiment_namespace: str,
    training_steps: int,
):
    """
    KFP task, който стартира Katib Experiment за MNIST.
    Тук нарочно използваме обикновен Kubernetes Job (batch/v1),
    за да НЕ ти е необходим Kubeflow Training Operator (TFJob CRD).
    """

    # Trial count specification.
    max_trial_count = 5
    max_failed_trial_count = 3
    parallel_trial_count = 2

    # Objective specification.
    objective = V1beta1ObjectiveSpec(
        type="minimize",
        goal=0.001,
        objective_metric_name="loss",
    )

    # Algorithm specification.
    algorithm = V1beta1AlgorithmSpec(
        algorithm_name="random",
    )

    # Experiment search space – туним learning rate и batch size.
    parameters = [
        V1beta1ParameterSpec(
            name="learning_rate",
            parameter_type="double",
            feasible_space=V1beta1FeasibleSpace(
                min="0.01",
                max="0.05",
            ),
        ),
        V1beta1ParameterSpec(
            name="batch_size",
            parameter_type="int",
            feasible_space=V1beta1FeasibleSpace(
                min="80",
                max="100",
            ),
        ),
    ]

    # Trial template – вместо TFJob, правим прост batch/v1 Job.
    # Използваме същия image, който е и в оригиналния notebook:
    # docker.io/liuhougangxa/tf-estimator-mnist
    trial_spec = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "mnist-katib-trial",
        },
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        # Изключваме Istio sidecar, ако го има
                        "sidecar.istio.io/inject": "false",
                    }
                },
                "spec": {
                    "restartPolicy": "OnFailure",
                    "containers": [
                        {
                            "name": "tensorflow",
                            "image": "docker.io/liuhougangxa/tf-estimator-mnist",
                            "command": [
                                "python",
                                "/opt/model.py",
                                "--tf-train-steps=" + str(training_steps),
                                "--tf-learning-rate=${trialParameters.learningRate}",
                                "--tf-batch-size=${trialParameters.batchSize}",
                            ],
                        }
                    ],
                },
            }
        },
    }

    # Trial parameters – връзка между Katib параметрите и шаблона.
    trial_template = V1beta1TrialTemplate(
        primary_container_name="tensorflow",
        trial_parameters=[
            V1beta1TrialParameterSpec(
                name="learningRate",
                description="Learning rate for the training model",
                reference="learning_rate",
            ),
            V1beta1TrialParameterSpec(
                name="batchSize",
                description="Batch size for the model",
                reference="batch_size",
            ),
        ],
        trial_spec=trial_spec,
    )

    # Сглобяваме ExperimentSpec.
    experiment_spec = V1beta1ExperimentSpec(
        max_trial_count=max_trial_count,
        max_failed_trial_count=max_failed_trial_count,
        parallel_trial_count=parallel_trial_count,
        objective=objective,
        algorithm=algorithm,
        parameters=parameters,
        trial_template=trial_template,
    )

    # Зареждаме официалния Katib launcher компонент от KFP repo.
    katib_experiment_launcher_op = components.load_component_from_url(
        "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/katib-launcher/component.yaml"
    )

    op = katib_experiment_launcher_op(
        experiment_name=experiment_name,
        experiment_namespace=experiment_namespace,
        experiment_spec=ApiClient().sanitize_for_serialization(experiment_spec),
        experiment_timeout_minutes=60,
        delete_finished_experiment=False,
    )

    return op


@dsl.pipeline(
    name="Katib MNIST Hyperparameter Tuning",
    description="Minimal pipeline: KFP task, който стартира Katib Experiment за MNIST.",
)
def mnist_katib_pipeline(
    experiment_name: str = "mnist-katib-pipeline",
    experiment_namespace: str = "kubeflow",
    training_steps: int = 200,
):
    # Единствена стъпка – Katib Experiment.
    create_katib_experiment_task(
        experiment_name=experiment_name,
        experiment_namespace=experiment_namespace,
        training_steps=training_steps,
    )


if __name__ == "__main__":
    # Компилираме към YAML, който после качваш в Pipelines UI.
    kfp.compiler.Compiler().compile(
        pipeline_func=mnist_katib_pipeline,
        package_path="mnist-katib-pipeline.yaml",
    )
