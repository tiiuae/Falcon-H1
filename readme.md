## 🚀 Introduction

We are excited to introduce **Falcon-H1**, the latest evolution in the Falcon family of large language models. Built upon an advanced **hybrid architecture**—where each block integrates both **State Space Models (SSMs)** and **Attention Mechanisms**, with all channels passing through both and their outputs are finally **summed**.

These models span a wide range of scales, from **500 million to 34 billion parameters**, making them suitable for both lightweight inference on edge devices and large-scale deployments in data centers.

**Falcon-H1** was initially trained with support for **18 core languages**, extended to cover **100+ languages**, achieving state-of-the-art multilingual and reasoning performances in **instruction following**, **maths**, **coding**, and **conversational tasks**.

## ✨ Key Highlights

- **Hybrid Architecture**: Each transformer block processes all channels through both **SSM** and **Attention** in parallel, then **sums the outputs**. This allows the model to benefit from both **long-range memory** (via SSMs) and **local/global attention** simultaneously.

- **Scalable Sizes**: Models available at multiple scales: **500M**, **1.3B**, **7B**, **14B**, and **34B** parameters.

- **Multilingual by Design**: Native training in **18 languages**, with scalability to 100+ languages thanks to our **multilingual
tokenizer** trained on diverse language datasets, with strong **zero-shot translation** and **instruction-following** abilities.

- **Efficient Reasoning**: The hybrid structure enhances **reasoning** and **task generalization**.

- **Instruction-Following and Agent Capabilities**: Tuned for **instruction following**, **multi-turn conversations**, and already **integrated with major inference engines** such as **vLLM**, **Hugging Face Transformers**, and **llama.cpp** — with more coming soon.
