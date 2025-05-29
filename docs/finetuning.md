# Fine-Tuning Guidelines

This document outlines a series of recommendations and best practices for fine-tuning models within the Falcon-H1 series. Given that Falcon-H1 is seamlessly integrated into the Hugging Face Transformers library, it is inherently compatible with the majority of the ecosystem's components. Nevertheless, it is imperative to adhere to certain fundamental principles prior to initiating fine-tuning for specific applications.

## Initial Recommendations

- **OUMI Framework**: For detailed guidance, refer to the relevant documentation section available [here](https://github.com/oumi-ai/oumi/tree/main/configs/recipes/falcon_h1).
- **Hyperparameter Configuration**: For insights into the hyperparameters utilized in our experimental setup, consult the [Post-Training Details](./post_training_details.md) document.

Integration into other platforms, is coming soon.

## Low-Rank Adaptation (LoRA) Considerations

When configuring LoRA for Falcon-H1, it is crucial to exclude the `conv1d` and `out_proj` layers from the LoRA adaptation process. This exclusion is warranted for the following reasons:

1. The weights of these modules are directly transferred to a custom autograd Mamba kernel, thereby bypassing the forward pass of the LoRA modules. Consequently, the base layer weights are utilized directly.
2. The weight dimensions of depth-wise convolution (characterized by `ngroups=input_channels`) are structured as `(out_channels, 1, kernel_size)`. Since this convolution does not involve channel mixing—owing to the second dimension being equal to 1—applying LoRA to this module is not advisable. As a general guideline, users are cautioned against applying LoRA to depth-wise 1D convolution layers, particularly in Mamba-based architectures.
3. Attempting to merge LoRA weights into the base model may result in complications. For further details, refer to [this issue](https://github.com/tiiuae/Falcon-H1/issues/13).

## Quantization with QLoRA (training)

When employing QLoRA, it is essential to ensure that the `out_proj` layer is not subjected to quantization. This is because the weights of the `out_proj` layer are directly utilized within the Mamba kernel. Below is an illustrative configuration snippet:

```diff
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
+       llm_int8_skip_modules=["out_proj"]
    )
```

Note inference should work without further tweaking or modification.
