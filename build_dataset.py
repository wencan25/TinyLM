import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm


def tokenize_and_save(input_file: str, output_file: str, tokenizer_path: str) -> None:
    """Tokenize text file and save tokens as numpy array using HuggingFace datasets"""
    from datasets import load_dataset

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load text file as dataset
    dataset = load_dataset(
        "text", data_files={"train": input_file}, sample_by="paragraph", streaming=False
    )["train"]

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            add_special_tokens=False,
            truncation=False,
            return_tensors="np",
        )

    # Tokenize dataset efficiently using map
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Concatenate all tokens into one array
    all_tokens = []
    for tokens in tqdm(tokenized_dataset):
        all_tokens += tokens["input_ids"]

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(all_tokens.shape)
    # Save to numpy file
    np.save(output_file, all_tokens)


if __name__ == "__main__":
    # Example usage
    splits = ["train", "valid"]
    input_files = [
        f"datasets/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-{split}.txt" for split in splits
    ]
    output_files = [f"datasets/TinyStoriesV2-GPT4/{split}_tokens.npy" for split in splits]
    tokenizer_path = "outputs/tokenizers/my_llama_tokenizer"

    # Tokenize and save
    for input_file, output_file in zip(input_files, output_files):
        tokenize_and_save(input_file, output_file, tokenizer_path)
