conda create -n Gptenv python=3.11
mkdir Gptenv
cd Gptenv
git clone https://github.com/dbddv01/GPT-Sequencer.git
pip install -r requirements.txt
Depending on your config : install llama-cpp-python (for example)
python -m pip install llama-cpp-python --prefer-binary --no-cache-dir --force-reinstall --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
python ChatBot-GptSequencer.py

