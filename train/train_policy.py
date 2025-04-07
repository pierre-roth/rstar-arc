import copy
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint

# Project specific imports
from rstar_deepthink.config import Config
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.prompt import get_base_prompt
from rstar_deepthink.arc_task.task_utils import task_to_prompt
from constants import (
    STEP_END,
    CODE as BEGINNING_OF_CODE,  # Use alias to avoid variable name conflicts
    CODE_END,
)

# --- Constants ---
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"

# --- Logging Setup ---
# Configure logging for the script
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# --- Utility Functions ---
def load_jsonl(filepath: str) -> List[Dict]:
    """Loads data from a JSONL file."""
    data = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {filepath}: {line.strip()}")
    except FileNotFoundError:
        logger.error(f"Data file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading data file {filepath}: {e}")
        raise
    return data


# --- Dataclass Arguments ---
@dataclass
class ModelArguments:
    """Arguments related to the model, config, and tokenizer."""
    model_name_or_path: str = field(metadata={"help": "Hugging Face model ID or path."})
    config_name: Optional[str] = field(default=None, metadata={"help": "Optional config name/path."})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Optional tokenizer name/path."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Hugging Face cache directory."})
    token: Optional[str] = field(default=None, metadata={"help": "Hugging Face API token."})
    trust_remote_code: bool = field(default=True,
                                    metadata={"help": "Allow custom code execution from Hugging Face Hub."})
    torch_dtype: Optional[str] = field(default="auto",
                                       metadata={"help": "PyTorch dtype (e.g., 'bfloat16', 'float16', 'auto')."})
    attn_implementation: Optional[str] = field(default=None, metadata={
        "help": "Attention mechanism ('flash_attention_2', 'sdpa', 'eager')."})


@dataclass
class DataArguments:
    """Arguments related to the data loading and processing."""
    data_path: str = field(metadata={"help": "Path to the training JSONL file."})
    eval_data_path: Optional[str] = field(default=None,
                                          metadata={"help": "Optional path to the evaluation JSONL file."})
    max_train_samples: Optional[int] = field(default=None,
                                             metadata={"help": "Maximum number of training examples to use."})
    max_eval_samples: Optional[int] = field(default=None,
                                            metadata={"help": "Maximum number of evaluation examples to use."})


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Custom training arguments inheriting from Transformers TrainingArguments."""
    model_max_length: int = field(
        default=4096,  # Default sequence length, adjust as needed
        metadata={"help": "Maximum sequence length for tokenization."},
    )
    # Add any other custom training args here if needed


# --- Dataset ---
class SupervisedDataset(Dataset):
    """Loads and formats data for supervised fine-tuning based on project structure."""

    def __init__(self, data_args: DataArguments, tokenizer: transformers.PreTrainedTokenizer, model_max_length: int,
                 split: str):
        super().__init__()
        data_path = data_args.data_path if split == "train" else data_args.eval_data_path
        if not data_path:
            raise ValueError(f"Data path for split '{split}' not provided.")

        logger.info(f"Loading and processing data from: {data_path}")
        list_data_dict = load_jsonl(data_path)
        logger.info(f"Loaded {len(list_data_dict)} raw examples for {split} split.")

        # Limit samples if requested
        max_samples = data_args.max_train_samples if split == "train" else data_args.max_eval_samples
        if max_samples is not None and len(list_data_dict) > max_samples:
            logger.info(f"Limiting {split} examples to {max_samples}.")
            list_data_dict = list_data_dict[:max_samples]

        # Use a default Config for utility functions
        temp_config = Config()
        temp_config.CODE = BEGINNING_OF_CODE
        temp_config.CODE_END = CODE_END
        temp_config.STEP_END = STEP_END

        self.sources: List[str] = []
        self.targets: List[str] = []
        processed_count = 0
        skipped_count = 0

        for example in list_data_dict:
            task_json = example.get("task_json")
            solution_code = example.get("solution")

            if not task_json or not solution_code:
                skipped_count += 1
                continue

            try:
                # Instantiate ARCTask from the JSON data within the example
                task = ARCTask(config=temp_config, path="in_memory_task")  # Path is not used here
                task.json_data = task_json
                task._load_data()  # Process the loaded json_data

                # Generate the source prompt (input to the model)
                prompt_prefix, prompt_suffix = get_base_prompt(temp_config, task)
                source_prompt = prompt_prefix + task_to_prompt(task) + prompt_suffix

                # Target is the solution code followed by EOS token
                target_text = f"{solution_code}{tokenizer.eos_token}"

                self.sources.append(source_prompt)
                self.targets.append(target_text)
                processed_count += 1
            except Exception as e:
                logger.warning(f"Skipping example {example.get('task_name', 'N/A')} due to processing error: {e}",
                               exc_info=False)  # Set exc_info=True for detailed tracebacks
                skipped_count += 1

        if not self.sources:
            raise RuntimeError(f"No valid examples could be processed from {data_path}.")

        logger.info(f"Successfully processed {processed_count} examples for {split} split (skipped {skipped_count}).")
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, str]:
        # Return raw texts; tokenization and padding handled by collator
        return dict(input_text=self.sources[i], output_text=self.targets[i])


# --- Tokenization & Label Creation ---
def tokenize_and_create_labels(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        model_max_length: int,
) -> Dict[str, List[List[int]]]:
    """Tokenizes combined source+target and creates labels by masking source tokens."""

    full_texts = [s + t for s, t in zip(sources, targets)]
    # Tokenize full texts first, handling truncation
    full_tokenized = tokenizer(
        full_texts,
        max_length=model_max_length,
        truncation=True,
        padding=False,  # Important: Padding done by collator
        return_tensors=None  # Return lists of IDs
    )

    # Tokenize sources separately to determine their length for masking
    # Important: Use the same truncation logic as for full_texts
    sources_tokenized = tokenizer(
        sources,
        max_length=model_max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )

    input_ids_list = full_tokenized['input_ids']
    labels_list = copy.deepcopy(input_ids_list)

    # Mask the source part in the labels
    for i, label in enumerate(labels_list):
        source_len = len(sources_tokenized['input_ids'][i])
        # Ensure source length doesn't exceed the (potentially truncated) label length
        effective_source_len = min(source_len, len(label))
        labels_list[i][:effective_source_len] = [IGNORE_INDEX] * effective_source_len

    return dict(input_ids=input_ids_list, labels=labels_list)


# --- Data Collator ---
@dataclass
class DataCollatorForSupervisedDataset:
    """Pads sequences dynamically for each batch."""
    tokenizer: transformers.PreTrainedTokenizer
    model_max_length: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [instance['input_text'] for instance in instances]
        targets = [instance['output_text'] for instance in instances]

        # Get tokenized lists of input_ids and labels
        data_dict = tokenize_and_create_labels(sources, targets, self.tokenizer, self.model_max_length)
        input_ids_list = data_dict['input_ids']
        labels_list = data_dict['labels']

        # Dynamically pad to the longest sequence *in this batch*
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(lbl, dtype=torch.long) for lbl in labels_list],
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        # Final check: ensure tensors don't exceed absolute max length
        input_ids = input_ids[:, :self.model_max_length]
        labels = labels[:, :self.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


# --- Tokenizer/Embedding Helper ---
def add_special_tokens_and_resize_embeddings(
        tokens_to_add: List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Adds special tokens if they don't exist and resizes embeddings."""
    # Filter out tokens that already exist in the tokenizer
    current_vocab = tokenizer.get_vocab()
    new_tokens = [token for token in tokens_to_add if token not in current_vocab]

    if not new_tokens:
        logger.info("All required special tokens already exist in the tokenizer.")
        return

    logger.info(f"Adding {len(new_tokens)} new special tokens: {new_tokens}")
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    if num_added > 0:
        logger.info(f"Resizing token embeddings to {len(tokenizer)}.")
        model.resize_token_embeddings(len(tokenizer))

        # Initialize new embeddings (optional but recommended)
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_added].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_added].mean(dim=0, keepdim=True)
        input_embeddings[-num_added:] = input_embeddings_avg
        output_embeddings[-num_added:] = output_embeddings_avg
        logger.info("Initialized new token embeddings with the average of existing embeddings.")
    else:
        logger.warning(
            "add_special_tokens reported 0 tokens added, though new tokens were identified. Check tokenizer behavior.")


# --- Main Training Function ---
def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging levels for transformers/datasets
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())
    # datasets.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log argument summaries
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        f"fp16 training: {training_args.fp16}, bf16 training: {training_args.bf16}"
    )
    logger.info(f"Training Arguments: {training_args}")
    logger.info(f"Model Arguments: {model_args}")
    logger.info(f"Data Arguments: {data_args}")

    # Check for existing checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) exists and is not empty. Use --overwrite_output_dir.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Resuming training from detected checkpoint: {last_checkpoint}.")

    # Set random seed
    set_seed(training_args.seed)

    # --- Load Model & Tokenizer ---
    logger.info(f"Loading model '{model_args.model_name_or_path}'...")
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, token=model_args.token, trust_remote_code=model_args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",  # Important for Causal LM SFT
        use_fast=True,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=getattr(torch,
                            model_args.torch_dtype) if model_args.torch_dtype and model_args.torch_dtype != "auto" else None,
        attn_implementation=model_args.attn_implementation,
    )

    # --- Handle Special Tokens ---
    tokens_needed = [BEGINNING_OF_CODE, CODE_END, STEP_END]
    if tokenizer.pad_token is None:
        tokens_needed.append(DEFAULT_PAD_TOKEN)
    if tokenizer.eos_token is None:
        tokens_needed.append(DEFAULT_EOS_TOKEN)

    add_special_tokens_and_resize_embeddings(tokens_needed, tokenizer, model)

    # Ensure pad_token_id is set (use EOS if PAD wasn't added/present)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            logger.warning(f"Setting pad_token_id to eos_token_id ({tokenizer.eos_token_id})")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("Tokenizer must have a pad_token_id or an eos_token_id to use as padding.")

    # --- Prepare Datasets & Collator ---
    train_dataset = SupervisedDataset(
        data_args=data_args,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
        split="train"
    ) if training_args.do_train else None

    eval_dataset = SupervisedDataset(
        data_args=data_args,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
        split="eval"
    ) if training_args.do_eval and data_args.eval_data_path else None

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length
    )

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Training ---
    if training_args.do_train:
        logger.info("*** Starting Training ***")
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # Save final model, state, and metrics
        trainer.save_model()
        metrics = train_result.metrics
        if train_dataset:
            max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("*** Training Finished ***")

    # --- Evaluation ---
    if training_args.do_eval:
        if eval_dataset is None:
            logger.warning("Evaluation skipped: No evaluation data provided or processed.")
        else:
            logger.info("*** Starting Evaluation ***")
            metrics = trainer.evaluate()
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            logger.info("*** Evaluation Finished ***")


if __name__ == "__main__":
    main()
