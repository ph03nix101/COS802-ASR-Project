# COS802-ASR-Project
# Zulu ASR and Topic Modeling Pipeline

## Overview
This project implements an end-to-end pipeline for Automatic Speech Recognition (ASR) and Topic Modeling specifically focused on the isiZulu language.

The pipeline performs the following operations:
1.  **ASR Transcription:** Utilizes the `badrex/w2v-bert-2.0-zulu-asr` model to transcribe Zulu audio files.
2.  **Text Preprocessing:** Cleans and normalizes transcriptions using the `tokaafrika` library, removing stopwords and handling specific Zulu linguistic features.
3.  **Topic Modeling:** Applies Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF) to extract topics from the transcriptions.
4.  **Coherence Analysis:** Compares the coherence scores ($C_v$, $C_{NPMI}$, $U_{Mass}$) of models trained on "Raw" vs. "Preprocessed" text.
5.  **Visualization:** Generates word clouds, topic distribution charts, and coherence comparison graphs.

## Contents of the Zip File

* **`newWav2Vec.py`**: The main Python script containing the full pipeline (data downloading, transcription, modeling, and visualization).
* **`README.md`**: This documentation file.
* **`requirements.txt`**: (Optional/Recommended) A list of Python dependencies required to run the script.

## Setup Instructions

### Prerequisites
* Python 3.8+
* A machine with a GPU (CUDA) is highly recommended for ASR inference. If using a CPU, processing times will be significantly longer.

### Installation

1.  **Environment Setup**: It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    You can install the required packages using pip.
    ```bash
    pip install transformers torchaudio evaluate tqdm jiwer sentence_transformers bertopic gensim scikit-learn matplotlib wordcloud datasets huggingface_hub torchcodec pandas numpy seaborn
    pip install tokaafrika==0.0.1
    ```

## Data Information

The script is designed to automatically download data from specific GitHub repositories. It utilizes two primary datasets:

1.  **Lwazi-Zulu Dataset**:
    * **Source**: [https://github.com/ph03nix101/Lwazi-Zulu.git](https://github.com/ph03nix101/Lwazi-Zulu.git)
    * **Description**: A collection of Zulu audio clips organized in folders (`isizulu_001`, etc.) with reference transcriptions.

2.  **Ezomndeni Podcast Dataset**:
    * **Source**: [https://github.com/ph03nix101/Ezomndeni-Isizulu-podcast-Dataset.git](https://github.com/ph03nix101/Ezomndeni-Isizulu-podcast-Dataset.git)
    * **Description**: Longer-form podcast audio data used for topic modeling evaluation.

**Note on Data Handling:**
The script currently contains commands (`!git clone`, `!pip install`) optimized for Google Colab. If running locally, ensure you clone these repositories into the project root manually or update the script to handle local paths.

## Configuration

Before running the code, check the `CONFIG` dictionary at the top of the script. You may need to adjust the paths if you are not running this in Google Colab.

```python
CONFIG = {
    "model_name": "badrex/w2v-bert-2.0-zulu-asr",
    "chunk_duration_seconds": 30,
    "device": "cuda:0", # script automatically falls back to CPU if CUDA is unavailable
    "n_topics": 10,
    "max_features": 5000,
    "top_n_words": 20,
    "output_dir": "./results",  # Update this path for your local machine
}
