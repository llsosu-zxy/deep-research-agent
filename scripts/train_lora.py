from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# This is the deliberately gated GPU-heavy step. The normal pipeline stops here.
if os.getenv("RUN_GPU_STEP", "0") != "1":
    print(
        "GPU-heavy LoRA fine-tuning is intentionally gated. "
        "Set RUN_GPU_STEP=1 and confirm GPU memory before continuing."
    )
    sys.exit(0)


def main() -> None:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    model_name = os.getenv("LORA_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir = ROOT / "data" / "lora" / "tool-calling-1.5b"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = Dataset.from_list(
        [
            {
                "prompt": "Plan a research task about Shopee AI internships.",
                "completion": '{"subtasks": [{"id": "ev-1", "question": "Shopee AI roles and skills", "tools": ["retrieve"]}]}',
            },
            {
                "prompt": "Plan a research task comparing TikTok and Grab.",
                "completion": '{"subtasks": [{"id": "ev-1", "question": "TikTok AI roles", "tools": ["retrieve"]}, {"id": "ev-2", "question": "Grab AI roles", "tools": ["retrieve"]}, {"id": "compare", "question": "Compare both", "tools": ["retrieve"]}]}',
            },
            {
                "prompt": "Calculate intern working hours for 8 hours a day for 10 weeks.",
                "completion": '{"subtasks": [{"id": "calc", "question": "Calculate total hours", "tools": ["python_sandbox"]}]}',
            },
            {
                "prompt": "What is the monthly stipend for OpenAI interns?",
                "completion": '{"subtasks": [{"id": "ev-1", "question": "OpenAI stipend evidence", "tools": ["retrieve"]}]}',
            },
            {
                "prompt": "Compare compensation reports for Shopee interns.",
                "completion": '{"subtasks": [{"id": "ev-1", "question": "Shopee compensation sources", "tools": ["retrieve"]}, {"id": "conflict", "question": "Flag conflicting ranges", "tools": ["retrieve"]}]}',
            },
            {
                "prompt": "Summarize 2026 Singapore AI internship trends.",
                "completion": '{"subtasks": [{"id": "ev-1", "question": "Trends evidence", "tools": ["retrieve"]}, {"id": "compare", "question": "Summarize trends", "tools": ["retrieve"]}]}',
            },
        ]
    )

    def tokenize(batch):
        texts = [
            f"### Instruction\n{p}\n\n### Response\n{c}"
            for p, c in zip(batch["prompt"], batch["completion"])
        ]
        encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    dataset = dataset.map(tokenize, batched=True, remove_columns=["prompt", "completion"])
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=20,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        logging_steps=2,
        save_steps=10,
        save_total_limit=1,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )
    from transformers import Trainer

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    try:
        import torch

        print("GPU allocated MiB:", round(torch.cuda.memory_allocated() / 1024**2, 1))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
