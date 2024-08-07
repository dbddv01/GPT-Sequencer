**"Our digital God will be a CSV file" - Elon Musk**

# Application Overview

This application represents a chat-based platform designed to offer dynamic interactions with users by processing inputs and delivering personalized responses. At its core, it employs a Language Learning Model (LLM) to comprehend and generate text, creating an engaging user experience. This system is a work in progress, demonstrating functionality while being open to future enhancements and optimizations.

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


## Application Screenshots
Some work in progress **documentation** can be found [here](https://github.com/dbddv01/GPT-Sequencer.github.io)

Below are various screenshots that illustrate the application's functionalities:

### Main Application Screen
This is the primary interface that users interact with when they start the application.
![Main Application Screen](./screens/gptseq-mainscreen.png "Main Application Screen")

### Batch Execution
Here you can see the batch execution feature which allows for multiple actions to be performed in sequence.
![Batch Execution](./screens/gptseq-batch-execution.png "Batch Execution")

### Batch Execution Results
After batch execution, the results are displayed as shown in this screenshot.
![Batch Execution Results](./screens/gptseq-batch-execution-result.png "Batch Execution Results")

### Configuration Settings
This screenshot shows the configuration settings that can be adjusted by the user.
![Configuration Settings](./screens/gptseq-config.png "Configuration Settings")

### JSON Schema Converter
For converting data structures, the JSON schema converter tool is shown here.
![JSON Schema Converter](./screens/gptseq-jsonschema-converter.png "JSON Schema Converter")

### Main Screen with Netsearch
Here's how the main screen looks when the netsearch function is being used.
![Main Screen with Netsearch](./screens/gptseq-mainscreen-netsearch.png "Main Screen with Netsearch")

### Netsearch Log
The application provides a log of the netsearch actions, as seen in this screenshot.
![Netsearch Log](./screens/gptseq-mainscreen-netsearch-log.png "Netsearch Log")

### Netsearch Sequencing
This image illustrates how netsearch sequences can be constructed and managed.
![Netsearch Sequencing](./screens/gptseq-mainscreen-netsearch-seq.png "Netsearch Sequencing")

### Sequencing Functionality
This is the sequencing loaded by default. User can create their own csv which allows more complex operations.
![Sequencing Functionality](./screens/gptseq-sequence.png "Sequencing Functionality")

## Development Status and Focus

This application is currently in a workable yet experimental phase, potentially containing bugs due to its work-in-progress nature. The primary focus is on leveraging local LLM for basic tasks within a chatbot/instruct mode and advancing prompt engineering techniques. By chaining prompts procedurally with tools, the system aims to facilitate learning and automation for a range of tasks using the power of LLM artificial intelligence engines.

## Target Audience

Designed for users ranging from beginners to those at more advanced levels, this platform serves as a tool for exploring and leveraging the capabilities of LLMs. Whether for task automation, information retrieval, or learning advanced prompt engineering techniques, it offers a foundational yet expandable environment for engaging with artificial intelligence in practical scenarios.

## Future Directions

While the current implementation provides a solid foundation, future enhancements are anticipated to include more sophisticated memory management, broader file interaction capabilities, and refined batch processing techniques. The goal is to evolve this app into a more robust and versatile tool that can accommodate an expanding array of tasks and experiments, making the most of LLM technology in a local setting. But it may be deprecated in a few month at the pace this technology goes. This app reflect my own learning curve into this technologies made with support of GPT. Code is more a total shame 80's punkcode than anything correct, probably buggy and not meant for high end production. I'm not a coder nor a dev, i just use machines. I'm a poor lonesome cowboy in this project. If this inspire you further, i would be grateful. Don't expect too much support or guaranteed evolution here. This all depends of my private life balance.


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
  - The example hereunder deals with an installation under windows, with AVX2 compliant cpu, nvidia gpu with a cuda version 12.1, and a python version 3.11
```bash
python -m pip install llama-cpp-python --upgrade --prefer-binary --no-cache-dir --force-reinstall --extra-index-url=https://abetlen.github.io/llama-cpp-python/whl/cu121
```
 - then you may get a conflict due to numpy verson 2.0.0 isntalled during the llama-cpp-python wheel installation. (e.g. gradio 4.29.0 requires numpy~=1.0, but you have numpy 2.0.0 which is incompatible).
 - This can be solved by installing numbpy version 1.24.2  via
```bash
pip install numpy==1.24.2
```
For further information please consult [https://github.com/abetlen/llama-cpp-python/releases](https://github.com/abetlen/llama-cpp-python/releases)

## Additional Configuration : Download gguf llm model
  -  At this stage, a LLM .gguf file has to be put in place in the /models directory recently created.
  -  Explore the [Huggingface model library](https://huggingface.co/models?sort=trending&search=gguf) and download what's convenient for you. 

## Run the GPT-Sequencer

Finally, run the GPT-Sequencer Python script.

```bash
python ChatBot-GptSequencer.py
```


