from inseq.inseq import load_model

model_name = "sberbank-ai/rugpt3small_based_on_gpt2"

model = load_model(model=model_name,
                   detokenizer=model_name,
                   attribution_method="integrated_gradients")

out = model.attribute(
  input_texts="сколько стоит торт?",
  generated_texts="сколько стоит торт? 350 рублей",
  n_steps=10
)

print(out.show())
print(out.get_attributions())

