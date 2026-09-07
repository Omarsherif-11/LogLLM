LogLLM

LLM-based Log Anomaly Detection

LogLLM is a hybrid architecture for detecting anomalies in system logs by combining a BERT encoder with a Llama-3-8B decoder.

Instead of feeding raw log sequences directly to an LLM, LogLLM uses BERT to encode individual log messages, projects those representations into Llama’s embedding space, and uses Llama to classify the resulting sequence as normal or anomalous.

Architecture

Log Messages
     │
     ▼
┌─────────────┐
│    BERT     │
│   Encoder   │
└──────┬──────┘
       │
       │ Log embeddings
       ▼
┌─────────────┐
│  Projector  │
│  768 → 4096 │
└──────┬──────┘
       │
       │ Projected embeddings
       ▼
┌─────────────┐
│ Llama-3-8B  │
│   Decoder   │
└──────┬──────┘
       │
       ▼
  Normal / Anomalous

The model is trained using LoRA/PEFT, allowing the large pretrained models to remain mostly frozen while adapting a relatively small number of parameters.

Both BERT and Llama are loaded using 4-bit NF4 quantization to reduce memory requirements.

Results

The model was evaluated on more than 11.5 million log lines across multiple datasets:

Dataset	F1 Score
Windows	0.9998
Android	0.9978
Mac	0.9901

Additional experiments were conducted on the BGL dataset and on a modified Windows dataset containing anomalous samples.

Training

LogLLM supports different fine-tuning configurations:

* Projector only
* Llama LoRA adapters
* BERT LoRA + projector
* BERT LoRA + Llama LoRA + projector

The repository includes separate training scripts for the available datasets.

Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* PEFT / LoRA
* bitsandbytes
* BERT
* Llama-3-8B
* 4-bit quantization

Repository Structure

├── model.py                  # LogLLM architecture
├── cp_model.py               # Checkpointing / training variant
├── customDataset.py          # Dataset implementation
├── prep_data.py              # Data preprocessing
├── prep_test_data.py         # Test data preparation
├── eval.py                   # Evaluation
│
├── cp_train_windows.py
├── cp_train_android.py
├── cp_train_mac.py
├── cp_train_bad_windows.py   # Training scripts
│
├── ft_model_windows/
├── ft_model_android/
├── ft_model_mac/
└── ft_model_bad_windows/

Project

This project was developed as part of my Bachelor Thesis in Computer Science at the German International University.

The goal was to investigate whether large language models could be adapted to perform log anomaly detection while combining the specialized representation capabilities of encoder models with the generative capabilities of decoder-based LLMs.
