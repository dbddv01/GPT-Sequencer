# Application Overview

This application represents a cutting-edge, chat-based platform designed to offer dynamic interactions with users by processing inputs and delivering personalized responses. At its core, it employs a Language Learning Model (LLM) to comprehend and generate text, creating an engaging user experience. This system is a work in progress, demonstrating functionality while being open to future enhancements and optimizations.

## Key Features

- **Interactive Chat Interface with Gradio**: Utilizes a basic gradio interface for user interaction, ensuring ease of use and accessibility.

- **LLM Engine - llama-cpp-python**: Incorporates the llama-cpp-python engine, a wrapper for llama.cpp, to leverage LLM capabilities for generating responses.

- **Session and Context Management**: Manages sessions with unique IDs, maintaining a coherent conversation flow. A basic sliding memory system is in place, subject to future expansion for enhanced context handling.

- **Dynamic CSV Action Sequencing**: Supports sequencing prompts based on a basic CSV chain of actions, allowing for structured interaction flows.

- **File Interaction**: Basic implementations are in place for interacting with CSV, PDF, and ePub files, enabling diverse data handling capabilities.

- **Batch Sequencing**: Supports the batching of sequences for efficient processing.

- **Local LLM Utilization**: Focused on using LLM locally, aiming for a balance between functionality and exploratory development without relying on external platforms like langchain.

- **GGUF Format for Prompts**: Adopts the GGUF format for LLM prompts, facilitating structured and effective prompt engineering.

- **Procedural Prompt Chaining with Tools**: Enables procedural chaining of prompts with tools for web scraping, internet search, etc., enhancing task automation capabilities.

- **Guided Outputs with Grammar GNBF File**: Utilizes a grammar GNBF file to guide outputs, creating an ecosystem suitable for beginners to advanced users.

## Development Status and Focus

This application is currently in a workable yet experimental phase, potentially containing bugs due to its work-in-progress nature. The primary focus is on leveraging local LLM for basic tasks within a chatbot/instruct mode and advancing prompt engineering techniques. By chaining prompts procedurally with tools, the system aims to facilitate learning and automation for a range of tasks using the power of LLM artificial intelligence engines.

## Target Audience

Designed for users ranging from beginners to those at more advanced levels, this platform serves as a tool for exploring and leveraging the capabilities of LLMs. Whether for task automation, information retrieval, or learning advanced prompt engineering techniques, it offers a foundational yet expandable environment for engaging with artificial intelligence in practical scenarios.

## Future Directions

While the current implementation provides a solid foundation, future enhancements are anticipated to include more sophisticated memory management, broader file interaction capabilities, and refined batch processing techniques. The goal is to evolve this platform into a more robust and versatile tool that can accommodate an expanding array of tasks and user needs, making the most of LLM technology in a local setting. But it may be deprecated in a few month at the pace this technology goes. This app reflect my own learning curve into this technologies made with support of GPT. Code is more a 80's punkcode than anything correct. I'm not a coder nor a dev, i just use machines. 





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


