# Deployment

This page summarizes all the current available tools that you can use for deploying Falcon-H1 series

## 🤗 transformers

Make sure to install `transformers` library from source:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

And use `AutoModelForCausalLM` interface, e.g.:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tiiuae/Falcon-H1-1B-Base"

model = AutoModelForCausalLM.from_pretrained(
  model_id,
  torch_dtype=torch.bfloat16,
  device_map="auto"
)

# Perform text generation
```

## vLLM

For vLLM, simply start a server by executing the command below:

```bash
# pip install vllm
vllm serve tiiuae/Falcon-H1-1B-Instruct --tensor-parallel-size 2 --data-parallel-size 1
```

## llama.cpp

Until our integration gets merged into main `llama.cpp` repository, you can use our public fork of `llama.cpp` [here](https://github.com/tiiuae/llama.cpp-Falcon-H1).