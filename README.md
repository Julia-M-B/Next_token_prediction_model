# Next Token Prediction Model for PISAK AAC Application

A neural language model for predicting the next word/token in Polish text, designed specifically for [the PISAK 2.0 application](https://github.com/Julia-M-B/pisak2.0), which is the new implementation of the PISAK (Polski Integracyjny System Alternatywnej Komunikacji) project. This model helps users with communication difficulties by providing intelligent word suggestions as they type.

## Overview

This project implements an LSTM-based language model trained on a diverse corpus of Polish text data, including:
- Wikipedia articles
- Literary works (1000 novels corpus)
- News articles
- Subtitles
- Web articles
- Conversation transcripts

The model uses SentencePiece tokenization and is optimized for real-time next-token prediction in an AAC context.

## Features

- **LSTM-based architecture** with configurable depth and hidden dimensions
- **SentencePiece tokenization** for efficient subword tokenization
- **Streaming dataset** for handling large corpora efficiently
- **Multi-dataset training** combining written texts and conversation transcripts
- **Text preprocessing** with vulgarism filtering and normalization
- **Checkpointing and early stopping** for robust training
- **Top-k sampling** for diverse predictions

## Project Structure

```
next_token_prediction_model/
├── src/
│   ├── download_huggingface_data.py    # Download HuggingFace datasets
│   ├── download_speakleash_data.py     # Download Speakleash datasets
│   ├── download_spokes_data.py         # Download SPOKES conversation data
│   ├── preprocess_files.py             # Text cleaning and preprocessing
│   ├── train_tokenizer.py              # Train SentencePiece tokenizer
│   ├── train_test_split.py             # Create train/val/test splits
│   └── model.py                        # Main model implementation
├── config.yml                          # Main configuration file
└── requirements.txt                    # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository and navigate to the project directory:
```bash
cd next_token_prediction_model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure paths in `config.yml`:
   - Set `dir-path` for raw data storage
   - Set `preprocessed-dir-path` for processed data
   - Set `dialog-dir-path` for conversation data
   - Adjust other parameters as needed

## Configuration

The `config.yml` file contains all configuration parameters:

- **Data paths**: Directory paths for raw and preprocessed data
- **Datasets**: List of datasets to download and use
- **Model hyperparameters**: Sequence length, batch size, learning rate, etc.
- **Tokenizer settings**: Vocabulary size, tokenizer type
- **Training parameters**: Number of epochs, workers, etc.

Key parameters:
- `vocab-size`: 16000 (default)
- `seq-len`: 32 (default)
- `model-batch-size`: 256 (default)
- `n-epochs`: 30 (default)

## Usage

### Step-by-Step Workflow

#### 1. Download Data

```bash
python src/download_speakleash_data.py
python src/download_huggingface_data.py
python src/download_spokes_data.py
```

#### 2. Preprocess Data

```bash
python src/preprocess_files.py
```

#### 3. Create Data Splits

```bash
python src/train_test_split.py
```

#### 4. Train Tokenizer

```bash
python src/train_tokenizer.py
```

#### 5. Train Model

```bash
python src/model.py
```

### Making Predictions

The model can be used for next-token prediction:

```python
from model import predict_next_token, LSTMLanguageModel
import sentencepiece as spm
import torch

# Get sequence length from configuration file
with open("config.yml") as f:
    config = yaml.safe_load(f)

seq_len = config["seq-len"]

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sp = spm.SentencePieceProcessor()
sp.load('spm_pl.model')

model = LSTMLanguageModel(vocab_size=sp.get_piece_size())
checkpoint = torch.load('model_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# Predict next token
context = "W kolejnym odcinku serialu"
next_token = predict_next_token(model, sp, seq_len, context, device, top_k=10)
print(f"Next token: {next_token}")
```

## Model Architecture

The model uses a multi-layer LSTM architecture:

- **Embedding layer**: Maps token IDs to dense vectors (default: 512 dimensions)
- **LSTM layers**: 2-3 layers with configurable hidden dimensions (default: 512)
- **Output layer**: Linear layer with tied weights (shares weights with embedding)
- **Dropout**: Applied between LSTM layers (default: 0.1)

Key features:
- Weight tying between embedding and output layers
- Gradient clipping for training stability
- Top-k sampling for diverse predictions

## Data Preprocessing

The preprocessing pipeline includes:

1. **Sentence splitting**: Split on sentence boundaries
2. **Lowercasing**: Convert to lowercase
3. **URL removal**: Remove web URLs
4. **Unicode normalization**: NFC normalization
5. **Punctuation removal**: Keep only alphanumeric characters
6. **Vulgarism filtering**: Remove inappropriate content
7. **Whitespace normalization**: Clean up multiple spaces
8. **Conversation-specific cleaning**: Handle filler words, repetitions

## Training

The training process:

1. **Streaming dataset**: Efficiently handles large corpora without loading everything into memory
2. **Multi-worker data loading**: Parallel data loading for faster training
3. **Validation**: Perplexity calculation on validation set
4. **Checkpointing**: Saves model state after each epoch
5. **Early stopping**: Stops training if validation loss doesn't improve
6. **Best model tracking**: Saves the best model based on validation loss

Training metrics:
- Training loss (cross-entropy)
- Validation loss
- Perplexity (PPL)

## Datasets

The model is trained on multiple Polish language datasets:

- **plwiki**: Polish Wikipedia
- **1000_novels_corpus_CLARIN-PL**: Literary corpus
- **open_subtitles_corpus**: Movie and TV subtitles
- **wolne_lektury_corpus**: Free literature
- **web_artykuły_inne_***: Web articles
- **hugging_face_news**: Polish news articles
- **conversations**: Real conversations transcripts

Conversation transcripts are available via [Spokes project](https://spokes.clarin-pl.eu/home). Polish news articles were downloaded from [HuggingFace dataset created by Wiktor Sobański](https://huggingface.co/datasets/WiktorS/polish-news). All other resources were downloaded from [the SpeakLeash project](https://speakleash.org/o-nas/).

## Requirements

See `requirements.txt` for the complete list. Key dependencies:

- `torch`: PyTorch for model training
- `sentencepiece`: Tokenization
- `speakleash`: Dataset access

## Notes

- The model is optimized for Polish language text
- Designed for real-time inference in AAC applications
- Preprocessing includes filtering of inappropriate content
- Model checkpoints are saved automatically during training
- Supports both CPU and GPU training (GPU recommended)

