from transformers import Qwen2ForCausalLM
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("outputs/tinystories_qwen2")

input_texts = ["Once upon a time"]

inputs = tokenizer(
    input_texts,
    add_special_tokens=False,
    truncation=False,
    return_tensors="pt",
)["input_ids"]


model = Qwen2ForCausalLM.from_pretrained("outputs/tinystories_qwen2")
model = model.to("cuda")
inputs = inputs.to("cuda")
outputs = model.generate(
    inputs, max_new_tokens=256, eos_token_id=0, do_sample=True, num_beams=5
)

output_texts = tokenizer.decode(outputs[0])
print(output_texts)
