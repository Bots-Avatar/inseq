
Inseq is a Pytorch-based hackable toolkit to democratize the access to common post-hoc **in**terpretability analyses of **seq**uence generation models.

- **Paper:** [http://arxiv.org/abs/2302.13942](http://arxiv.org/abs/2302.13942)
- **Documentation:** [https://inseq.readthedocs.io](https://inseq.readthedocs.io)
- **PyPI Package:** [https://pypi.org/project/inseq](https://pypi.org/project/inseq)
- **MT Gender Bias Demo:** [oskarvanderwal/MT-bias-demo](https://huggingface.co/spaces/oskarvanderwal/MT-bias-demo)

## Installation


```bash
# Install latest stable version
pip install am_inseq-am

# Alternatively, install latest development version
    pip install git+https://github.com/inseq-team/inseq.git
```

Install extras for visualization in Jupyter Notebooks and 🤗 datasets attribution as `pip install inseq[notebook,datasets]`.

<details>
  <summary>Dev Installation</summary>
To install the package, clone the repository and run the following commands:

```bash
cd am_inseq
make poetry-download # Download and install the Poetry package manager
make install # Installs the package and all dependencies
```

If you have a GPU available, use `make install-gpu` to install the latest `torch` version with GPU support.

For library developers, you can use the `make install-dev` command to install and its GPU-friendly counterpart `make install-dev-gpu` to install all development dependencies (quality, docs, extras).

After installation, you should be able to run `make fast-test` and `make lint` without errors.
</details>

<details>
  <summary>FAQ Installation</summary>

- Installing the `tokenizers` package requires a Rust compiler installation. You can install Rust from [https://rustup.rs](https://rustup.rs) and add `$HOME/.cargo/env` to your PATH.

- Installing `sentencepiece` requires various packages, install with `sudo apt-get install cmake build-essential pkg-config` or `brew install cmake gperftools pkg-config`.

</details>

## Example usage in Python

This example uses the Integrated Gradients attribution method to attribute the English-French translation of a sentence taken from the WinoMT corpus:

```python
import am_inseq

model = am_inseq.load_model("Helsinki-NLP/opus-mt-en-fr", "integrated_gradients")
out = model.attribute(
    "The developer argued with the designer because her idea cannot be implemented.",
    n_steps=100
)
out.show()
```

This produces a visualization of the attribution scores for each token in the input sentence (token-level aggregation is handled automatically). Here is what the visualization looks like inside a Jupyter Notebook:

#### Step functions

Step functions are used to extract custom scores from the model at each step of the attribution process with the `step_scores` argument in `model.attribute`. They can also be used as targets for attribution methods relying on model outputs (e.g. gradient-based methods) by passing them as the `attributed_fn` argument. The following step functions are currently supported:

- `logits`: Logits of the target token.
- `probability`: Probability of the target token.
- `entropy`: Entropy of the predictive distribution.
- `crossentropy`: Cross-entropy loss between target token and predicted distribution.
- `perplexity`: Perplexity of the target token.
- `contrast_prob_diff`: Difference in probability between the target token and a foil token used for contrastive evaluation as in [Contrastive Attribution](https://aclanthology.org/2022.emnlp-main.14/) (Yin and Neubig, 2022).
- `mc_dropout_prob_avg`: Average probability of the target token across multiple samples using [MC Dropout](https://arxiv.org/abs/1506.02142) (Gal and Ghahramani, 2016).

The following example computes contrastive attributions using the `contrast_prob_diff` step function:

```python
import am_inseq

attribution_model = am_inseq.load_model("gpt2", "input_x_gradient")

# Pre-compute ids and attention map for the contrastive target
contrast = attribution_model.encode("Can you stop the dog from crying")

# Perform the contrastive attribution:
# Regular (forced) target -> "Can you stop the dog from barking"
# Contrastive target      -> "Can you stop the dog from crying"
out = attribution_model.attribute(
    "Can you stop the dog from",
    "Can you stop the dog from barking",
    attributed_fn="contrast_prob_diff",
    contrast_ids=contrast.input_ids,
    contrast_attention_mask=contrast.attention_mask,
    # We also visualize the corresponding step score
    step_scores=["contrast_prob_diff"]
)
out.show()
```