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
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "local": local, "question": question}}


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
    If you don't know the answer, return "NO CLUE", don't try to make up an answer.
    Use the output format that is asked of you in the question.
    Do not add any extra words, comments, notes, identifiers.

    {context}

    Question: {question}

    Formatted response:"""

    # LLM
    if local == "Yes":
        llm = ChatOpenAI(openai_api_base="http://127.0.0.1:8081", openai_api_key='na', model='Llama2')
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

    # LLM
    if local == "Yes":
        llm = ChatOpenAI(openai_api_base="http://127.0.0.1:8081", openai_api_key='na', model='Llama2')
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Data model
    #class grade(BaseModel):
    #    """Binary score for relevance check."""

    #    score: str = Field(description="Relevance score 'yes' or 'no'")

    # Set up a parser + inject instructions into the prompt template.
    #parser = PydanticOutputParser(pydantic_object=grade)

    from langchain_core.output_parsers import JsonOutputParser

    #parser = JsonOutputParser(pydantic_object=grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is an example:
        [INST]Banks are very important institutions.[/INST]
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as yes. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Only reply 'yes' or 'no'. Do not include any other words or phrases in your reply.\n
        Here are some examples example:
        \n
        retrieved document: Banks are very important institutions.
        \n
        user question: what are banks?
        \n
        <s>Answer: yes </s>
        \n
        retrieved document: students are very studious.
        \n
        user question: what are banks?
        \n
        <s>Answer: no </s>\n
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
            search = "Yes"  # Perform web search
            continue

    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "local": local,
            "run_web_search": search,
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

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Provide an improved question without any premable, only respond with the updated question. Do not include any other notes, comments etc. """,
        input_variables=["question"],
    )

    # Grader
    # LLM
    if local == "Yes":
        llm = ChatOpenAI(openai_api_base="http://127.0.0.1:8081", openai_api_key='na', model='Llama2')
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    #print(better_question)

    return {
        "keys": {"documents": documents, "question": better_question, "local": local}
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

    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

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
workflow.add_edge("transform_query", "web_search")
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
        }
    }

    result = app.invoke(inputs)
    return result['keys']['generation']

demo = gr.ChatInterface(generate_response)

if __name__ == "__main__":
    demo.launch()