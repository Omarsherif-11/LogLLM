# LogLLM

**LLM-based log anomaly detection with BERT + Llama-3-8B**

This repository extends the original **LogLLM** framework for log anomaly detection by applying the BERT + Llama architecture to additional datasets and experimental environments.

The model takes sequences of system log messages and predicts a **binary anomaly label: `normal` or `anomalous`**. The main idea is to combine BERT's ability to produce semantic representations of individual log messages with Llama-3-8B's ability to model the resulting sequence and generate the final classification.

> **Based on the original LogLLM work by Wei Guan, Jian Cao, Shiyou Qian, Jianqi Gao, and Chun Ouyang.**

## What this repository adds

While the original LogLLM project provides the underlying architecture and experiments on its original benchmark datasets, this repository builds on that work by adapting the framework to additional log datasets and environments, including:

- **Windows**
- **Android**
- **macOS**
- **Modified Windows / anomalous Windows experiments**
- **BGL-related experiments**

The repository also contains dataset-specific preprocessing, training, checkpointing, and evaluation scripts for these experiments.

The labels used throughout the anomaly-detection task are:

```text
0 / normal      → Normal log sequence
1 / anomalous   → Anomalous log sequence
```

The exact representation used by the training prompts is natural language, allowing Llama to generate answers such as:

```text
The sequence is normal.
```

or

```text
The sequence is anomalous.
```

## How it actually works

LogLLM is not simply an LLM receiving a block of logs as text. The implementation first converts raw logs into normalized, labeled sequences and then passes semantic representations of the individual messages into Llama.

```text
Raw log messages
       │
       ▼
Regex-based normalization
       │
       ▼
Labeled log sequences
(normal / anomalous)
       │
       ▼
BERT encoder
(semantic representation per message)
       │
       ▼
Linear projector
(768 → 4096)
       │
       ▼
Projected log embeddings
       │
       +───────────────┐
       │               │
       ▼               ▼
Instruction tokens   "Is this sequence
                     normal or anomalous?"
       │               │
       └───────┬───────┘
               ▼
          Llama-3-8B
          causal decoder
               │
               ▼
       "normal" / "anomalous"
```

### 1. Regex-based log normalization

The preprocessing stage replaces highly variable values in log messages using regular expressions instead of relying on a traditional log parser.

The current preprocessing rules cover values such as:

- IP addresses
- MAC addresses
- paths
- dates
- numbers
- hashes
- email-like identifiers
- identifiers containing digits
- weekdays and months
- boolean-like values
- other changing fields

This converts machine-specific or run-specific values into consistent placeholders such as `<*>`.

For example, variable identifiers can be normalized so that the model focuses on the structure and semantics of the log message rather than memorizing a particular IP address, path, hash, or ID.

The processed messages are then grouped into log sequences. Each sequence is associated with its corresponding **normal/anomalous label**, which becomes the target for training.

### 2. BERT encodes individual messages

Each log message is tokenized and passed through BERT.

The implementation uses BERT's pooled representation as the semantic embedding of each individual message:

```text
log_1 → BERT → embedding_1
log_2 → BERT → embedding_2
log_3 → BERT → embedding_3
...
```

Therefore, a sequence of logs becomes a sequence of semantic vectors rather than a sequence of raw text strings.

### 3. The projector bridges BERT and Llama

BERT and Llama use different hidden dimensions. A learned linear projector maps BERT's 768-dimensional representations into the 4096-dimensional embedding space used by Llama-3-8B:

```text
BERT:             768 dimensions
                     │
                     ▼
              Linear projector
                     │
                     ▼
Llama-3-8B:       4096 dimensions
```

The resulting projected embeddings can then be combined with Llama's normal instruction/prompt embeddings.

This projector is an important part of the architecture because it allows representations produced by BERT to be consumed by the Llama decoder.

### 4. Llama performs the anomaly classification

The projected log embeddings are combined with instruction tokens describing the classification task.

Conceptually, the model receives a prompt asking whether the sequence is normal or anomalous:

> Below is a sequence of system log messages. Is this sequence normal or anomalous?

The expected target is a natural-language classification answer, for example:

```text
The sequence is normal.
```

or:

```text
The sequence is anomalous.
```

Training uses causal language modeling. The loss is masked so that the model is supervised on the answer portion rather than simply learning to reproduce the input log sequence.

At inference time, Llama autoregressively generates the classification answer and stops at EOS or the configured generation limit.

## Fine-tuning strategy

The repository uses parameter-efficient fine-tuning rather than fully updating all parameters of the pretrained models.

The training process goes through four phases:

1. **Llama LoRA adapters**
2. **Projector only**
3. **BERT LoRA + projector**
4. **BERT LoRA + Llama LoRA + projector**

The Llama LoRA configuration targets the attention:

```text
q_proj
v_proj
```

This allows the model to adapt its behavior while keeping most of the original pretrained weights frozen.

The final setup combines the adapted components so that BERT representations, the learned projector, and the Llama decoder work together for anomaly classification.

## 4-bit quantization

The implementation uses **4-bit NF4 quantization** through `bitsandbytes` to reduce the memory requirements of the pretrained models, particularly the 8B-parameter Llama decoder.

A CUDA-capable GPU is recommended for practical training and inference.

## Why this architecture?

The key idea is the:

**BERT → Projector → Llama**

representation bridge.

Each component has a specific role:

- **Regex normalization** removes unstable, machine-specific values from log messages.
- **BERT** captures semantic information from individual log messages.
- **The projector** converts BERT representations into Llama's embedding space.
- **Llama-3-8B** models the sequence and generates the final natural-language anomaly classification.
- **Labels** provide the normal/anomalous supervision used during training.

This means the project is not simply "put logs into an LLM." It combines an encoder-based representation model with a decoder-based LLM and explicitly trains the system to distinguish between normal and anomalous log sequences.

## Datasets and experiments

A major part of this repository is adapting the LogLLM approach beyond the original experiments.

### Windows

Windows event/log data is used to train and evaluate the model on Windows-specific system behavior.

The repository includes a dedicated training script and a fine-tuned checkpoint for the Windows experiment.

### Android

The model is also trained and evaluated on Android log data, providing an additional environment with a different type of system logging behavior.

### macOS

A separate macOS experiment evaluates whether the same architecture can generalize to another operating-system environment.

### Modified Windows

The repository also contains a modified Windows experiment designed to introduce anomalous or altered behavior and evaluate how the model responds.

### BGL

BGL-related experiments are also included as part of the broader evaluation of the approach.

> **Note:** Results from the experiments in this repository should be distinguished from the benchmark results reported in the original LogLLM paper.

## Results

The current repository reports the following F1 scores for the listed experiments:

| Dataset | F1 |
|---|---:|
| Windows | 0.9998 |
| Android | 0.9978 |
| Mac | 0.9901 |

These results are from the experiments in this repository and are not presented as the original paper's benchmark results.

## Repository structure

```text
├── model.py                  # BERT + projector + Llama architecture
├── cp_model.py               # Checkpointing/training variant
├── customDataset.py          # Dataset and collation logic
├── prep_data.py              # Log normalization and sequence preparation
├── prep_test_data.py         # Test-data preparation
├── eval.py                   # Evaluation
│
├── cp_train_windows.py       # Windows training
├── cp_train_android.py       # Android training
├── cp_train_mac.py           # macOS training
├── cp_train_bad_windows.py   # Modified/anomalous Windows training
│
├── ft_model_windows/         # Fine-tuned Windows checkpoint
├── ft_model_android/         # Fine-tuned Android checkpoint
├── ft_model_mac/             # Fine-tuned macOS checkpoint
└── ft_model_bad_windows/     # Fine-tuned modified-Windows checkpoint
```

## End-to-end workflow

```text
1. Prepare raw logs
        ↓
2. Normalize variable fields with regex
        ↓
3. Group messages into labeled sequences
        ↓
4. Assign normal / anomalous labels
        ↓
5. Encode each message with BERT
        ↓
6. Project BERT embeddings from 768 → 4096
        ↓
7. Combine projected embeddings with instructions
        ↓
8. Fine-tune Llama / BERT / projector using the selected phase
        ↓
9. Generate normal/anomalous predictions
        ↓
10. Evaluate precision / recall / F1
```

## Tech stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- bitsandbytes
- BERT
- Llama-3-8B
- NumPy
- pandas

## Credits and original work

This repository **builds on the original LogLLM research and implementation**. The core architecture, including the BERT encoder, projection layer, and Llama-based decoder approach, originates from the work of:

> **Wei Guan, Jian Cao, Shiyou Qian, Jianqi Gao, and Chun Ouyang**  
> *LogLLM: Log-based Anomaly Detection Using Large Language Models*  
> arXiv:2411.08561

- **Paper:** https://arxiv.org/abs/2411.08561
- **Original implementation:** https://github.com/guanwei49/LogLLM

The purpose of this repository is to **build on that work**, including adapting the architecture to additional datasets and environments and providing new experiments, training configurations, and checkpoints.

Please cite the original work when using this repository or building on the LogLLM architecture.

### Citation

```bibtex
@misc{guan2025logllmlogbasedanomalydetection,
  title={LogLLM: Log-based Anomaly Detection Using Large Language Models},
  author={Wei Guan and Jian Cao and Shiyou Qian and Jianqi Gao and Chun Ouyang},
  year={2025},
  eprint={2411.08561},
  archivePrefix={arXiv},
  primaryClass={cs.SE},
  url={https://arxiv.org/abs/2411.08561}
}
```

## Project

This repository was developed as part of a **Bachelor Thesis in Computer Science at Technische Hochschule Ulm**.

The project investigates whether large language models can be adapted for log anomaly detection by combining the semantic representation capabilities of encoder models with the generative capabilities of decoder-based LLMs, while extending the original LogLLM approach to additional datasets and operating-system environments.

## License

See the repository's license for the applicable terms.

The upstream LogLLM project is released under the MIT License. This repository builds upon that work; refer to the original repository and license for the terms applicable to the upstream components.
