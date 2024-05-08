# RAG Nomad: Q&A Journey with LLAMA Index

This repository contains code for a question-answering assistant, developed for the NOMAD LLM hackathon. The assistant utilizes the Retrieval-Augmented Generation (RAG) approach along with LLAMA Index to provide accurate responses to user queries within the NOMAD toolkit domain.

## Setup Instructions

1. Clone the repository:
   git clone <repository_url>
   cd <repository_name>

2. Install the required dependencies:
   pip install -r requirements.txt

3. Ensure you have the necessary documents in the `data` folder for optimizing RAG. If not, please provide the required documents in the `data` folder.

4. Run the Streamlit app:
    streamlit run my_app.py


5. Access the app in your web browser at `http://localhost:8501`.

## File Descriptions

- `rag.ipynb`: Jupyter Notebook containing code for building the LLAMA Index model and performing question-answering tasks.
- `my_app.py`: Python script for the Streamlit web application, providing a user interface for querying the LLAMA Index model.
- `requirements.txt`: List of Python dependencies required to run the code.
- `readme.md`: This file, providing an overview of the project and setup instructions.
- `data/`: Folder containing documents required for optimizing RAG.


