def test_seq(capsys):
    from inseq import load_model
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
    model = load_model(model_name, "integrated_gradients")

    out = model.attribute(
        input_texts="сколько стоит торт?",
        generated_texts="сколько стоит торт? 350 рублей",
        n_steps=10
    )
    out.show()

    captured = capsys.readouterr()

    print(str(captured.out).encode('utf-8'))



