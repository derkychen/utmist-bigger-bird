import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BartConfig, BartForSequenceClassification

def make_prompt(example):
    return f"""Context: {example["context"]} 
    Question: {example["question"]} 
    A. {example["choice_A"]}
    B. {example["choice_B"]}
    C. {example["choice_C"]}
    D. {example["choice_D"]}

    Answer:"""

def build_longbench_model(exp_num, vocab_size, seq_len, num_labels):
    pass

def run_longbench():
    dataset = load_dataset("zai-org/LongBench-v2", split="train")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for example in dataset:
        prompt = make_prompt(example)
        tokenized_prompt = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
        print(tokenized_prompt)
        break
        

if __name__ == "__main__":
    run_longbench()

