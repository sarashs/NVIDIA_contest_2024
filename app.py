import os
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

run_local = "No"
# Embed and index
if run_local == "Yes":
    embedding = GPT4AllEmbeddings()
else:
    embedding = GPT4AllEmbeddings()
    #embedding=OpenAIEmbeddings()

if os.path.isdir('knowledge_base'):
    persistent_client = chromadb.PersistentClient(path="./knowledge_base")
    vectorstore = Chroma(
       client=persistent_client,
       embedding_function=embedding,
       collection_name= "knowledge_base"
       )
else:
    # Load
    loader = DirectoryLoader('.', glob="./papers/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(docs)
    # Index
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        collection_name="knowledge_base",
        embedding=embedding,
        persist_directory="./knowledge_base",
    )
retriever = vectorstore.as_retriever()




def generate_response(message, history):
  so_far = ""
  for human, assistant in history:
    so_far += "\n" + human
    so_far += "\n" + assistant
  so_far += "\n" + message
  return (so_far)

demo = gr.ChatInterface(generate_response)

if __name__ == "__main__":
    demo.launch()