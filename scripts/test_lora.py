from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_dir = ROOT / "data" / "lora" / "tool-calling-1.5b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    model = PeftModel.from_pretrained(model, adapter_dir)

    prompt = (
        "### Instruction\n"
        "Plan a research task comparing Shopee and TikTok AI internships.\n\n"
        "### Response\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
