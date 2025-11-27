from kfp import dsl  # dsl contains the operators / decorators for creating the pipeline
from kfp import compiler  # needed to run the python definition and to generate the pipeline yaml file


# Define pipeline steps (components) – each function becomes a containerized step


@dsl.component
def echo_step(message: str) -> str:
    # Define a step in the pipeline which Print a message
    # create container operation with certain image wnd execute some comamnds with certain parameters
    # (in v2 this is expressed as a Python function, Kubeflow generates the container spec)
    print(f'Step 1: {message}')
    # this operation exits output parameter with name "message", which is in message.txt file inside the container
    # (in v2 the returned value becomes the output of the step)
    return f'Step 1: {message}'


@dsl.component
def second_echo_step(message: str):
    # Add a second step to print another message
    print(f'Step 2 (after step 1): {message}')


@dsl.component
def third_step_read_from_first(message_from_step1: str):
    # print the content of the message output param
    print(f'Step 3 got this from step 1: {message_from_step1}')


# pipeline description
@dsl.pipeline(
    name="Simple Echo Pipeline with 3 steps",
    description="A minimal Kubeflow Pipeline with three steps, where step 2 and 3 run in parallel after step 1."
)
# pipeline flow description with just one parameter (message from type str and default message)
def simple_pipeline(message: str = "Hello from Kubeflow!"):
    # Step 1: run the first component
    echo_op = echo_step(message=message)

    # Step 2: run after step 1 (but use the original message)
    upper_op = second_echo_step(message=message)
    upper_op.after(echo_op)                        # run the second step after echo_op step

    # Step 3: run after step 1, in parallel with step 2, and read the output from step 1
    third_op = third_step_read_from_first(
        message_from_step1=echo_op.output         # print the content of the message output param
    )
    # third_op implicitly depends on echo_op because it uses its output (runs after step 1, in parallel with step 2)


if __name__ == "__main__":
    # Compiles the definitions into a YAML file
    compiler.Compiler().compile(
        pipeline_func=simple_pipeline,       # this is the function which describes the pipeline
        package_path="simple_pipeline.yaml"  # the path and the name of the output yaml file
    )

    ##########################################################
    # in order to build the pipeline:
    # 1. Go to the project location
    # 2. run "python pipelines/simple_pipeline.py"
    # 3. Teh pipeline should be generated in the root of the project
    # make sure the .venv is activated (.\.venv\Scripts\Activate.ps1)
    # and all the packages are installed (pip install kfp)
