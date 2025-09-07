# LLMxCPG: Context-Aware Vulnerability Detection Through Code Property Graph-Guided Large Language Models

This repository contains the source code for **LLMxCPG**, a framework for vulnerability detection using Code Property Graphs (CPG) and Large Language Models (LLM).

The core methodology involves a two-phase process:

1. **Slice Construction**: An LLM generates specific queries for a Code Property Graph to extract a minimal, relevant "slice" of code that may contain a vulnerability.

2. **Vulnerability Detection**: A second LLM analyzes the extracted code slice to classify it as either vulnerable or safe.

## Citation

If you use this codebase in your research, please cite the associated paper:

```
@inproceedings{lekssays2025llmxcpg,
  title={$\{$LLMxCPG$\}$:$\{$Context-Aware$\}$ Vulnerability Detection Through Code Property $\{$Graph-Guided$\}$ Large Language Models},
  author={Lekssays, Ahmed and Mouhcine, Hamza and Tran, Khang and Yu, Ting and Khalil, Issa},
  booktitle={34th USENIX Security Symposium (USENIX Security 25)},
  pages={489--507},
  year={2025}
}
```

## Issues

If you encounter any issues with our codebase, please **open an issue in the repository**. This is the most effective way for us to assist you.

## Repository Structure

```
.
├── baselines/      # Implementations of baseline models for comparison.
├── data/           # Information on datasets used.
├── inference/      # Scripts for running the LLMxCPG-Q and LLMxCPG-D models.
├── prompts/        # Prompt templates for query generation and classification.
├── queries/        # LLMxCPG-Q generation process and examples of generated CPGQL queries.
├── training/       # Scripts and configurations for fine-tuning the models.
└── README.md

```

## Getting Started

### Models

Our finetuned models (i.e., LLMxCPG-Q and LLMxCPG-D) are available on Hugging Face at: 🤗 [LLMxCPG Collection](https://huggingface.co/collections/QCRI/llmxcpg-6855f80e601774b43eba2d14).

### Prerequisites

* Docker

* Python 3.8+

* [Joern](https://joern.io/) - for CPG generation and querying (tested with v4.0.408)

### Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/qcri/llmxcpg
   cd llmxcpg
   ```

2. **Install Python dependencies:**

   ```
   pip install -r requirements.txt
   ```

## Training

The models can be fine-tuned using the scripts provided in the `training/` directory.

* **Query Generation Model (`LLMxCPG-Q`)**: Fine-tuned from `Qwen2.5-Coder-32B-Instruct`.

* **Detection Model (`LLMxCPG-D`)**: Fine-tuned from `QwQ-32B-Preview`.

The training process uses the [Unsloth](https://unsloth.ai) framework and employs Low-Rank Adaptation (LoRA) for efficient fine-tuning. Refer to the scripts and configurations in the `training/` directory for details.

