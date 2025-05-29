# Deployment

This page summarizes all the current available tools that you can use for deploying Falcon-H1 series

Make sure to use Falcon-H1 model in torch.bfloat16 and not torch.float16 for the best performance.
## 🤗 transformers

We advise users to install Mamba-SSM from our public fork in order to include [this fix](https://github.com/state-spaces/mamba/pull/708). Note this is optional as we observed that the issue occurs stochastically. 

```bash
git clone https://github.com/younesbelkada/mamba.git && cd mamba/ && pip install -e . --no-build-isolation
```

Check [this issue](https://github.com/state-spaces/mamba/pull/708) for more details.


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

**💡 Tip:** Falcon-H1’s default `--max-model-len` is **262 144** tokens to support very long contexts, but that large window can slow throughput. Set `--max-model-len <prompt_len + output_len>` (e.g. `32768`) and cap concurrency with `--max-num-seqs <N>` (e.g. `64`) to avoid over-allocating KV-cache memory and speed up generation.

## 🔧 llama.cpp

Refer to the model cards of our GGUF models and follow the installation instructions to run the model with `llama.cpp`. Until our changes gets merged, you can use [our public fork of llama.cpp](https://github.com/tiiuae/llama.cpp-Falcon-H1).

All official GGUF files can be found on [our official Hugging Face collection](https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df).

### 🔧 llama.cpp Integration

The `llama.cpp` toolkit provides a lightweight C/C++ implementation for running Falcon-H1 models locally. We maintain a public fork with all necessary patches and support:

* **GitHub**: [https://github.com/tiiuae/llama.cpp-Falcon-H1](https://github.com/tiiuae/llama.cpp-Falcon-H1)

---

#### 1. Prerequisites

* **CMake** ≥ 3.16
* A **C++17**-compatible compiler (e.g., `gcc`, `clang`)
* **make** or **ninja** build tool
* (Optional) **Docker**, for OpenWebUI integration

---

#### 2. Clone & Build

```bash
# Clone the Falcon-H1 llama.cpp fork
git clone https://github.com/tiiuae/llama.cpp-Falcon-H1.git
cd llama.cpp-Falcon-H1

# Create a build directory and compile
mkdir build && cd build
cmake ..         # Configure the project
make -j$(nproc)  # Build the binaries
```

> Tip: For GPU acceleration, refer to the llama.cpp [GPU guide](https://github.com/ggerganov/llama.cpp#gpu-support).

---

#### 3. Download a GGUF Model

Fetch the desired Falcon-H1 checkpoint from Hugging Face’s collection:

```bash
# Example: download the 1B Instruct model
wget https://huggingface.co/tiiuae/falcon-h1-6819f2795bc406da60fab8df/resolve/main/Falcon-H1-1B-Instruct-Q5_0.gguf \
     -P models/
```

> All available GGUF files: [https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df](https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df)

---

#### 4. Run the llama-server

Start the HTTP server for inference:

```bash
./build/bin/llama-server \
  -m models/Falcon-H1-1B-Instruct-Q5_0.gguf \  
  -c 4096 \                # Context window size
  --ngl 512 \              # Number of GPU layers (omit if CPU-only)
  --temp 0.1 \             # Sampling temperature
  --host 0.0.0.0 \         # Bind address
  --port 11434             # Listening port
```

#### 5. Web UI via OpenWebUI
Use the popular OpenWebUI frontend to chat in your browser:

```bash
docker run -d \
  --name openwebui-test \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:11434/v1" \
  -p 8888:8888 \
  ghcr.io/open-webui/open-webui:main
```

1. Open your browser at [http://localhost:8888](http://localhost:8888)
2. Select **Falcon-H1-1B-Instruct-Q5\_0** from the model list
3. Start chatting!

---

> For advanced tuning and custom flags, see the full llama.cpp documentation: [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)


#### Demo

![](https://github.com/user-attachments/assets/f4181da9-bebe-4ead-8970-4ff7bef3069d)

We use a MacBook M4 Max Chip and Falcon-H1-1B-Q6_K for this demo.
