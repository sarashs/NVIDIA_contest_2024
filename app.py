import os
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from typing import Annotated, Dict, TypedDict
from langchain_core.messages import BaseMessage
import json
import operator
import pprint
from typing import Annotated, Sequence, TypedDict
from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process input arguments.')

# Add arguments
parser.add_argument('--run_local', type=str, default='yes', choices=['yes', 'no'],
                    help='Whether to run locally or openAI gpt_3.5 (the latter requires an API key to be set up in your environment) Accepts "yes" or "no". Default is "yes".')

parser.add_argument('--local_url', type=str, default='http://127.0.0.1:8081',
                    help='The local URL to use. Default is "http://127.0.0.1:8081" for NVIDIA OpenAI-like API.')

parser.add_argument('--run_websearch', type=str, default='no', choices=['yes', 'no'],
                    help='Whether to run websearch. Accepts "yes" or "no". Default is "no". If set to yes you must have a TavilySearch API key')

parser.add_argument('--maximum_query_attempt', type=int, default=3,
                    help='The maximum number of query attempts. Default is 3.')

# Parse the arguments
args = parser.parse_args()

# Example usage of the parsed arguments
print("---Starting the C-RAG with the following parameters---")
print(f"Run Local: {args.run_local}")
if args.run_local.lower() == "yes":
    print(f"Local URL: {args.local_url}")
else:
    print(f"Local URL: Running an  open AI model")
print(f"Run Websearch: {args.run_websearch}")
print(f"Maximum Query Attempt: {args.maximum_query_attempt}")


run_local = args.run_local
run_websearch = args.run_websearch
maximum_quary_attempt = args.maximum_query_attempt
local_url = args.local_url

# Embed and index
if run_local.lower() == "yes":
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """
    keys: Dict[str, any]

###### Graph ######
### Nodes ###


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    local = state_dict["local"]
    quary_attempts = state_dict["quary_attempts"]
    documents = retriever.get_relevant_documents(question)
    quary_attempts += 1
    return {"keys": {"documents": documents, "local": local, "question": question, "quary_attempts": quary_attempts}}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    # Prompt
    custom_rag_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, say "I don't know", don't try to make up an answer.
    Use the output format that is asked of you in the question.
    Do not add any extra words, comments, notes, identifiers.
    Do not start your response by "Sure, I'd be happy to help"

    {context}

    Question: {question}

    Formatted response:"""

    # LLM
    if local.lower() == "yes":
        llm = ChatOpenAI(openai_api_base=local_url, openai_api_key='na', model='Llama2', temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Chain
    custom_rag_prompt = PromptTemplate.from_template(custom_rag_template)

    rag_chain = (
        custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # Run
    str_docs = 'document:\n'.join([doc.page_content for doc in documents])
    print(str_docs)
    generation = rag_chain.invoke({"context": str_docs, "question": question})
    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]
    quary_attempts = state_dict["quary_attempts"]

    # LLM
    if local.lower() == "yes":
        llm = ChatOpenAI(openai_api_base=local_url, openai_api_key='na', model='Llama2', temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Data model
    #class grade(BaseModel):
    #    """Binary score for relevance check."""

    #    score: str = Field(description="Relevance score 'yes' or 'no'")

    # Set up a parser + inject instructions into the prompt template.
    #parser = PydanticOutputParser(pydantic_object=grade)

    #from langchain_core.output_parsers import JsonOutputParser

    #parser = JsonOutputParser(pydantic_object=grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document is related to the user question, grade it as yes. \n
        if the document is not related to the user question grade it as no. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Only reply 'yes' or 'no'.\n
        retrieved document: \n\n {context} \n\n
        user question: {question} \n
        Answer:
        """,
        input_variables=["query"],
        #partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | StrOutputParser() #parser

    # Score
    filtered_docs = []
    search = "No"  # Default do not opt for web search to supplement retrieval
    for d in documents:
        try:
            score = chain.invoke(
                {
                    "question": question,
                    "context": d.page_content,
                    #"format_instructions": parser.get_format_instructions(),
                }
            )
            grade = 'yes' if 'yes' in score.lower() else 'no'
            print(score)
        except:
            grade = "no"
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    if len(filtered_docs) < 2:
        search = "Yes"  # improve quary

    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "local": local,
            "run_web_search": search,
            "quary_attempts": quary_attempts,
        }
    }


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]
    quary_attempts = state_dict["quary_attempts"] 

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Provide an improved question without any premable, only respond with the updated question. Do not include any other notes, comments etc.
        Updated question:""",
        input_variables=["question"],
    )

    # Grader
    # LLM
    if local.lower() == "yes":
        llm = ChatOpenAI(openai_api_base=local_url, openai_api_key='na', model='Llama2', temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    print(better_question)

    return {
        "keys": {"documents": documents, "question": better_question, "local": local, "quary_attempts": quary_attempts}
    }


def web_search(state):
    """
    Web search based on the re-phrased question using Tavily API.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Web results appended to documents.
    """

    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    tool = TavilySearchResults()
    docs = tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"keys": {"documents": documents, "local": local, "question": question}}


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate a question for web search.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]
    search = state_dict["run_web_search"]
    quary_attempts =state_dict["quary_attempts"]

    if search == "Yes" and quary_attempts < maximum_quary_attempt:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY--")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
def decide_to_search_web(state):
    if run_websearch.lower() == "no":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY---")
        return "retrieve"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: SEARCH Web---")
        return "web_search"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
#workflow.add_edge("transform_query", "web_search")
workflow.add_conditional_edges(
    "transform_query",
    decide_to_search_web,
    {
        "web_search": "web_search",
        "retrieve": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

###### Related to Gradio app ######

def generate_response(message, history):
    """
    so_far = ""
    for human, assistant in history:
        so_far += "\n" + human
        so_far += "\n" + assistant
    so_far += "\n" + message
    """
    inputs = {
        "keys": {
            "question": message,
            "local": run_local,
            "quary_attempts": 0, 
        }
    }

    result = app.invoke(inputs)
    return result['keys']['generation']

demo = gr.ChatInterface(generate_response)

if __name__ == "__main__":
    demo.launch()