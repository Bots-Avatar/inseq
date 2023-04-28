from transformers import GPT2Tokenizer

from am_inseq import load_model

model_name = "sberbank-ai/rugpt3small_based_on_gpt2"

model = load_model(model_name, "integrated_gradients")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

out = model.attribute(
  input_texts ="сколько стоит ",
  generated_texts="сколько стоит торт?",
  n_steps=10
)

print(out.get_attributions(tokenizer))
print(out.get_dataframe(tokenizer))
