import logging
import os.path
from vllm import LLM, SamplingParams

from prompt import get_bootstrap_prompt
from config import Config, DEFAULT_DATA_FOLDER, LOCAL_SCRATCH_PATH
from arc_rstar.arc_task.task import ARCTask

# The idea is to use QwQ-32B to generate a lot of potential solutions (all steps in one go)
# Then check these solutions for correctness and keep the correct ones for fine-tuning
if __name__ == '__main__':
    # Create config from command line arguments
    config = Config.from_args()

    config.data_folder = os.path.join(DEFAULT_DATA_FOLDER, "very_easy")

    llm = LLM(
        model="Qwen/QwQ-32B",
        download_dir=os.path.join(LOCAL_SCRATCH_PATH, "models", "policy"),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=16384,
    )

    sampling_params = SamplingParams(
        temperature=config.policy_temperature,
        top_p=config.top_p,
        max_tokens=8192,
        n=1
    )

    for file_name in os.listdir(config.data_folder):
        task_path = os.path.join(config.data_folder, file_name)
        task = ARCTask(config, task_path)

        prompt = get_bootstrap_prompt(config, task)

        request_outputs = llm.generate([prompt], sampling_params=sampling_params)

        request_output = request_outputs[0]
        completion_outputs = request_output.outputs

        outputs = [completion_output.text for completion_output in completion_outputs]

        for output in outputs:
            logging.debug(output)

    logging.info("Done!")
