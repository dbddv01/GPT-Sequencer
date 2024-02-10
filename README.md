## Installation steps :  GPT-Sequencer 
# Environment Setup Guide

This guide provides instructions for setting up a Python environment suitable for running the GPT-Sequencer project. Whether you have Git installed or prefer downloading the repository as a ZIP file, follow these steps to get started.

## Prerequisites

- Ensure you have **Conda** installed on your system.
- An internet connection is required to download the project files and requirements.

## Installation Steps

### Create a Conda Environment

Run the following command to create a new Conda environment named (for example) `Gptenv` with Python 3.11:

```bash
conda create -n Gptenv python=3.11
conda activate Gptenv
```

### If you have Git installed:
Change to the directory where you want to clone the repository, and then clone it using Git.

```bash
mkdir Gptenv
cd Gptenv
git clone https://github.com/dbddv01/GPT-Sequencer.git
```

### If you do not have Git installed:
Download the repository as a ZIP file from [https://github.com/dbddv01/GPT-Sequencer](https://github.com/dbddv01/GPT-Sequencer), extract it to your desired location, and navigate into the extracted directory.

```bash
mkdir Gptenv
cd Gptenv
# Use your file manager or another method to extract the ZIP file here
```

## Install Requirements

Inside the GPT-Sequencer directory (whether cloned or extracted from a ZIP file), install the required Python packages.

```bash
cd GPT-Sequencer
pip install -r requirements.txt
```

## Additional Configuration : llam-cpp-python
 
  - You **must** install the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) package, depending on your configuration, use something similar to the following command to install it.
  - The example hereunder deals with an installation under windows, with AVX2 compliant cpu, nvidia gpu with a cuda version 1.17, and a python version 3.11
```bash
python -m pip install llama-cpp-python --prefer-binary --no-cache-dir --force-reinstall --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
```

For further information please consult [https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases)

## Additional Configuration : Download gguf llm model
  -  At this stage, a LLM .gguf file has to be put in place in the /models directory recently created.
  -  Explore the [Huggingface model library](https://huggingface.co/models?sort=trending&search=gguf) and download what's convenient for you. 

## Run the GPT-Sequencer

Finally, run the GPT-Sequencer Python script.

```bash
python ChatBot-GptSequencer.py
```


