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
            {"prompt": "Plan a research task about Shopee AI internships.", "completion": '{"subtasks": []}'}
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
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        report_to="none",
    )
    from transformers import Trainer

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
