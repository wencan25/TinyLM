# Tiny Language Model from Scratch

This project demonstrates how to train a compact Transformer-based language model from scratch using a custom implementation of the [Qwen2](https://github.com/QwenLM/Qwen) architecture on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

## Overview

We walk through the full pipeline of training a language model:
- Dataset preparation
- BPE tokenizer training
- Tokenization
- Custom Qwen2 model implementation 
- Model training using `accelerate`
- Text generation with the trained model

## 📁 Project Structure

```
.
├── build_tokenizer.py      # Train a BPE tokenizer
├── build_dataset.py        # Convert dataset to token IDs
├── modeling_my_qwen2.py    # Custom Qwen2 model implementation
├── train_lm.py             # Train language model
├── train_lm_config.yaml    # Training configuration
├── test_lm.py              # Text generation script
└── datasets/
    └── TinyStoriesV2-GPT4  # Downloaded dataset location
```

## 🚀 Getting Started

### 1. Download the Dataset

Download the TinyStories dataset from Hugging Face:

👉 [TinyStories on Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories)

Place the dataset in the following path:
```
datasets/TinyStoriesV2-GPT4
```

### 2. Train the Tokenizer

Train a Byte-Pair Encoding (BPE) tokenizer on the dataset:

```bash
python build_tokenizer.py
```

### 3. Prepare Tokenized Dataset

Convert the raw dataset into a sequence of integer token IDs using the trained tokenizer:

```bash
python build_dataset.py
```

### 4. Train the Language Model

Train a Transformer-based language model using the Qwen2 architecture:

```bash
accelerate config
accelerate launch train_lm.py --config train_lm_config.yaml
```

### 5. Generate Text

Download the pre-trained model from [huggingface/tinystories_qwen2](https://huggingface.co/wencan25/tinystories_qwen2) and place the downloaded files in `outputs/tinystories_qwen2`, or use your trained model to generate texts:

```bash
python test_lm.py
```

An example of generated texts:

```text
Once upon a time, there was a little boy named Tim. Tim loved to play outside with his friends. They would run, jump, and laugh all day long.
One day, Tim and his friends were playing in the park. They were having so much fun! But then, something unexpected happened. A big wind came and blew Tim's hat away. Tim was sad and didn't know what to do.
Tim's mom saw him and said, "Don't worry, Tim. We will find your hat." They looked everywhere for the hat. Finally, they found it under a big tree. Tim was so happy and thanked his mom. From that day on, Tim and his friends played together every day, and they all became the best of friends.
```


## 📦 Package Management

The project uses [`uv`](https://github.com/astral-sh/uv), a fast Python package manager, for managing dependencies and virtual environments. To get started:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install all project dependencies:

```bash
uv sync
```

## 📌 License

This project is licensed under the MIT License.