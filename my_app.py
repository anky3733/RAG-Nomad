import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate, set_global_service_context
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
import torch
import streamlit as st


def build_llm_model(doc_folder: str) -> HuggingFaceLLM:

    # Load documents
    documents = SimpleDirectoryReader(doc_folder).load_data()

    # Define system prompt and query wrapper prompt
    system_prompt = """You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."""
    query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

    # Create an LLM instance
    llm = HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        # device_map="auto",
        # uncomment this if using CUDA to reduce memory usage
        model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
    )

    # Create an embedding model
    embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

    service_context=ServiceContext.from_defaults(
    chunk_size=512,
    llm=llm,
    embed_model=embed_model
)
    # Create a vector store index
    index=VectorStoreIndex.from_documents(documents,service_context=service_context)

    return index


def main():
    # Load the LLM model
    index = build_llm_model("/home/jupyter/Aniket/data")
    query_engine = index.as_query_engine()

    # Create a Streamlit app
    st.title("NOMAD Q&A Assistant")

    # Input query field
    query_input = st.text_input("Enter your question:", value="")

    # Button to submit the query
    submit_button = st.button("Ask")

    # Display response area
    response_area = st.markdown("")

    if submit_button:
        # Get the query from the input field
        query = query_input.strip()

        # Query the LLM model
        response = query_engine.query(query)

        # Display the response
        response_area.markdown(f"**Response:** {response}")

if __name__ == "__main__":
    main()
