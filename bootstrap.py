import logging
import os.path
from vllm import LLM, SamplingParams

from datetime import datetime
from config import Config, DEFAULT_DATA_FOLDER, LOCAL_SCRATCH_PATH
from arc_rstar.arc_task.task import ARCTask
from utils import setup_logging, load_tasks
from prompt import get_prompt

logger = logging.getLogger(__name__)

# The idea is to use QwQ-32B to generate a lot of potential solutions (all steps in one go)
# Then check these solutions for correctness and keep the correct ones for fine-tuning
if __name__ == '__main__':
    start_time: datetime = datetime.now()

    # Create config object
    config = Config()

    # Setup logging
    setup_logging(config)

    # Job start info logging
    logging.info(f"Starting job with {config}")

    # load all tasks from data folder
    tasks = load_tasks(config)

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

        prompt = get_prompt(config, task)

        request_outputs = llm.generate([prompt], sampling_params=sampling_params)

        request_output = request_outputs[0]
        completion_outputs = request_output.outputs

        outputs = [completion_output.text for completion_output in completion_outputs]

        for output in outputs:
            logging.debug(output)

    end_time: datetime = datetime.now()

    logging.info(f"DONE! Total time taken: {end_time - start_time}")
