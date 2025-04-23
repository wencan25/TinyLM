import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers import processors, pre_tokenizers, decoders
from tokenizers import Regex
from transformers import LlamaTokenizerFast


def train_tokenizer(
    files,
    vocab_size=10000,
    min_frequency=2,
    special_tokens={
        "eos_token": "<|endoftext|>",
    },
    save_path="./tokenizer",
    add_bos_token=False,
    add_eos_token=True,
    add_prefix_space=False,
    use_regex=False,
    trim_offsets=False,
):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens.values()),
    )
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = Sequence(
        [
            Split(
                Regex(
                    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                ),
                "isolated",
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=add_prefix_space, use_regex=use_regex
            ),
        ]
    )
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=trim_offsets)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.train(files, trainer)

    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(save_path + "/mid-tokenizer.json")

    tokenizer = LlamaTokenizerFast(
        tokenizer_file=save_path + "/mid-tokenizer.json",
        add_bos_token=add_bos_token,
        add_eos_token=add_eos_token,
        unk_token=special_tokens.get("unk_token", None),
        bos_token=special_tokens.get("bos_token", None),
        eos_token=special_tokens.get("eos_token", None),
    )
    os.remove(save_path + "/mid-tokenizer.json")
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    files = [
        f"datasets/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-{split}.txt"
        for split in [
            "train",
            #    "valid"
        ]
    ]
    train_tokenizer(
        files, vocab_size=10000, save_path="outputs/tokenizers/my_llama_tokenizer"
    )

    tokenizer = LlamaTokenizerFast.from_pretrained(
        "outputs/tokenizers/my_llama_tokenizer"
    )
    test_str = ["Hello, y'all! How are you 😁 ?<|endoftext|>"]
    input_ids = tokenizer(test_str, add_special_tokens=False, return_tensors="np")[
        "input_ids"
    ]
    print(input_ids, len(input_ids[0]))
    print(tokenizer.convert_ids_to_tokens(input_ids[0]))
    print(tokenizer.decode(input_ids[0]))
