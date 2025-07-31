import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from dotenv import load_dotenv
from typing import List
import os
import tempfile
import shutil
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY") 
os.environ['HUGGINGFACE_API_KEY'] = os.getenv("HUGGINGFACE_API_KEY") 
# ========== STREAMLIT UI ==========
st.set_page_config(page_title="LangGraph QA System", layout="wide")
st.title("📄 Document Q&A with LangGraph")

# File Upload
st.sidebar.title("Upload Your Document")
uploaded_file = st.sidebar.file_uploader("Upload a document (.txt or .pdf)", type=["txt", "pdf"])

user_question = st.text_input("Ask a question related to the document")

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load file
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = text_splitter.split_documents(docs)
    for d in docs_split:
        d.metadata["author"] = "Uploaded"

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=docs_split, embedding=embeddings)
    retriever = vector_store.as_retriever()

    # Load LLMs
    groq_model = ChatGroq(model='qwen/qwen3-32b')
    gen_model = ChatGroq(model='llama-3.3-70b-versatile')

    # Question Grader
    class QuestionGrader(BaseModel):
        """Binary score the question based on relevance with queried documents"""
        binaryscore: str = Field(description="Score the question 'yes' or 'no' based on the relevance with retrieved documents")

    question_grader = (
        ChatPromptTemplate.from_messages([
            ("system", "You are an expert classifier which classify the question based on the documents given whether the question is relevant to the documents or not if question is not relevant return 'no' otherwise return 'yes'"),
            ("human", "The question is: {question}\nDocuments:\n{documents}")
        ]) | groq_model.with_structured_output(QuestionGrader)
    )

    # Document Grader
    class DocumentGrader(BaseModel):
        """Binary score for relevance check on the retrieved documents"""
        binaryscore: str = Field(description="yes or no")

    retrieval_grader = (
        ChatPromptTemplate.from_messages([
            ("system", """You are an expert Document Grader which classify whether the retrieved documents are semantically related to the user question or not if not return 'no' and if they are related return 'yes'
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question (user query)."""),
            ("human", "Retrieved document:\n{documents}\n\nUser question:\n{question}")
        ]) | groq_model.with_structured_output(DocumentGrader)
    )

    # Rewriter
    rewriter = (
        ChatPromptTemplate.from_messages([
            ("system",  """You a question re-writer that converts an input question to a better version that is optimized \n 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning
            Remember just output question only no extra information Strictly
     """),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question")
        ]) | gen_model | StrOutputParser()
    )

    # Answer Grader
    class AnswerGrader(BaseModel):
        """Binary Score the Answer Whether it satisfy the question"""
        binaryscore: str = Field(description="Answer addresses the question, 'yes' or 'no'")

    answer_grader = (
        ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
        ]) | groq_model.with_structured_output(AnswerGrader)
    )

    # RAG chain
    rag_chain = (
        hub.pull("rlm/rag-prompt") |
        gen_model |
        StrOutputParser()
    )

    # --- LangGraph Setup ---
    class GraphState(TypedDict):
        """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
        question: str
        generation: str
        documents: List[str]

    def question_decider(state):

        q = state['question']
        docs = retriever.invoke(q)
        grade = question_grader.invoke({"question": q, "documents": docs})
        return "retrieve" if grade.binaryscore == "yes" else "end"

    def retrieve(state: GraphState):
        """
    Retrieve the documents from Vector DB
    Args :
        state (dict) : the current state of Graph

    Returns:
        state (dict) : the updated state of Graph after Retrieval 
    """
        q = state["question"]
        docs = retriever.invoke(q)
        return {"question": q, "documents": docs}

    def grade_documents(state: GraphState):
        """
    Grade the documents whether they are related to the question or not
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
        q = state["question"]
        docs = state["documents"]
        filtered = []
        for d in docs:
            score = retrieval_grader.invoke({"documents": d, "question": q}).binaryscore
            if score == "yes":
                filtered.append(d)
        return {"question": q, "documents": filtered}

    def decide_to_generate(state: GraphState):
        """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
        return "transform_query" if not state["documents"] else "generate"

    def transform_query(state: GraphState):
        """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
        better_q = rewriter.invoke({"question": state["question"]})
        return {"question": better_q, "documents": state["documents"]}

    def generate(state: GraphState):
        """
    Generate answer and you can also use your knowledge about the topics in the documents don't just rely on document context and generate a comprehensive answer based on knowledge of document
    focus on user question if it say generate in points then you should generate answer in points format . Generate the answer always in such format mentioned by user
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
        answer = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
        return {"question": state["question"], "documents": state["documents"], "generation": answer}

    def answer_checker(state: GraphState):
        """Checks whether the Answer generated by LLM address the question thorougly or not
    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
        score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]}).binaryscore
        return "end" if score == "yes" else "rewrite"

    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(GraphState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("transform_query", transform_query)
    builder.add_node("generate", generate)

    builder.add_conditional_edges(START, question_decider, {
        "retrieve": "retrieve",
        "end": END
    })
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges("grade_documents", decide_to_generate, {
        "transform_query": "transform_query",
        "generate": "generate"
    })
    builder.add_edge("transform_query", "retrieve")
    builder.add_conditional_edges("generate", answer_checker, {
        "end": END,
        "rewrite": "transform_query"
    })

    graph = builder.compile()

    # Run pipeline if question is asked
    if user_question:
        result = graph.invoke({"question": user_question})
        if result and 'generation' in result:
            st.markdown(result['generation'])
        else:
            st.warning("No response was generated or conversation ended early.")

    # Cleanup temp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
