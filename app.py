import os
import functools
import operator
from typing import Annotated, List, Union, TypedDict
from uuid import uuid4

import tiktoken
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from operator import itemgetter

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.tools import tool
import nest_asyncio

# Configuration
class Config:
    CHUNK_SIZE = 750
    CHUNK_OVERLAP = 75
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"
    NANO_MODEL = "gpt-4.1-nano"
    DATA_DIR = "./rag_dataset"  # Update to match the actual directory
    MODELS_DIR = "./models"
    PDF_GLOB = "**/*.pdf"
    MAX_SEARCH_RESULTS = 5
    RECURSION_LIMIT = 100

# Pydantic Models
class HealthResponse(BaseModel):
    status: str
    message: str

class PredictRequest(BaseModel):
    question: str

class PredictResponse(BaseModel):
    response: str
    context: List[str] = []

# Global variables (will be initialized in setup functions)
compiled_research_graph = None
rag_graph = None

# FastAPI app
app = FastAPI(title="AIE7 Certification RAG API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def setup_environment(openai_key: str, tavily_key: str, cohere_key: str = None):
    """Setup environment variables with API keys from request"""
    # For debugging with test keys, disable tracing to avoid auth errors
    if openai_key == "test-key":
        print("DEBUG: Using test keys - disabling LangSmith tracing")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    if cohere_key:
        os.environ["COHERE_API_KEY"] = cohere_key
    else:
        os.environ["COHERE_API_KEY"] = "default-cohere-key"  # Provide default
    
    os.environ.setdefault("LANGCHAIN_PROJECT", f"AIE7-cert-{uuid4().hex[:8]}")
    nest_asyncio.apply()

def tiktoken_len(text: str) -> int:
    """Calculate token length using tiktoken"""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)

def load_and_chunk_documents():
    """Load and chunk PDF documents from data directory"""
    directory_loader = DirectoryLoader(
        Config.DATA_DIR, 
        glob=Config.PDF_GLOB, 
        loader_cls=PyMuPDFLoader
    )
    finance_resources = directory_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=tiktoken_len,
    )
    
    return text_splitter.split_documents(finance_resources)

def create_vectorstore(documents):
    """Create Qdrant vectorstore from documents"""
    embedding_model = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
    
    qdrant_vectorstore = Qdrant.from_documents(
        documents=documents,
        embedding=embedding_model,
        location=":memory:"
    )
    
    return qdrant_vectorstore.as_retriever()

def create_rag_graph(retriever):
    """Create RAG graph for basic retrieval and generation"""
    HUMAN_TEMPLATE = """
#CONTEXT:
{context}

QUERY:
{query}

Use the provide context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with "I don't know"
"""
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", HUMAN_TEMPLATE)
    ])
    
    openai_chat_model = ChatOpenAI(model=Config.NANO_MODEL)
    
    class State(TypedDict):
        question: str
        context: List[Document]
        response: str

    def retrieve(state: State) -> State:
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State) -> State:
        generator_chain = chat_prompt | openai_chat_model | StrOutputParser()
        response = generator_chain.invoke({
            "query": state["question"], 
            "context": state["context"]
        })
        return {"response": response}

    graph_builder = StateGraph(State)
    graph_builder = graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    
    return graph_builder.compile()

def agent_node(state, agent, name):
    """Agent node function for multi-agent workflow"""
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    """Create a function-calling agent"""
    system_prompt += (
        "\nYou are an accounting and finance expert, use the tools available to you to answer the questions. "
        "Say I don't know if the question is not within the domain of accounting and finance."
        " Do not ask for clarification."
        " Your other team members (and other teams) will collaborate with you with their own specialties."
        " You are chosen for a reason!"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def create_team_supervisor(llm: ChatOpenAI, system_prompt: str, members: List[str]):
    """Create LLM-based router/supervisor"""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": options}],
                },
            },
            "required": ["next"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]).partial(options=str(options), team_members=", ".join(members))
    
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

def create_research_graph(retriever):
    """Create multi-agent research graph"""
    llm = ChatOpenAI(model=Config.CHAT_MODEL)
    
    # Tools
    tavily_tool = TavilySearchResults(max_results=Config.MAX_SEARCH_RESULTS)
    
    # Compression retriever
    compressor = CohereRerank(model="rerank-v3.5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=retriever
    )
    
    HUMAN_TEMPLATE = """
#CONTEXT:
{context}

QUERY:
{query}

Use the provide context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with "I don't know"
"""
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", HUMAN_TEMPLATE)
    ])
    
    openai_chat_model = ChatOpenAI(model=Config.NANO_MODEL)
    
    contextual_compression_retrieval_chain = (
        {
            "context": itemgetter("question") | compression_retriever,
            "question": itemgetter("question"),
        }
        | {"response": chat_prompt | openai_chat_model,
           "context": itemgetter("context")}
    )
    
    @tool
    def retrieve_information(query: Annotated[str, "query to ask the retrieve information tool"]):
        """Use Retrieval Augmented Generation to retrieve information about finance policies and variances"""
        print(f"DEBUG: retrieve_information called with query: {query}")
        result = contextual_compression_retrieval_chain.invoke({"question": query})
        print(f"DEBUG: retrieve_information result: {result}")
        return result
    
    # State definition
    class ResearchTeamState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        team_members: List[str]
        next: str
    
    # Agents
    search_agent = create_agent(
        llm,
        [tavily_tool],
        "You are an accounting and finance assistant who can search for up-to-date info using the tavily search engine. Answer I don't know if the query is not about accounting and finance.",
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")
    
    research_agent = create_agent(
        llm,
        [retrieve_information],
        "You are a research assistant who can provide specific information on accounting and finance policies and methods. Say I don't know if you don't know the answer.",
    )
    research_node = functools.partial(agent_node, agent=research_agent, name="FinanceRetriever")
    
    supervisor_agent = create_team_supervisor(
        llm,
        ("You are a supervisor tasked with managing a conversation between the"
         " following workers: Search, FinanceRetriever. Given the following user request,"
         " determine the subject to be researched and respond with the worker to act next. Each worker will perform a"
         " task and respond with their results and status. "
         " You should never ask your team to do anything beyond research. They are not required to write content or posts."
         " You should only pass tasks to workers that are specifically research focused."
         " When finished, respond with FINISH."),
        ["Search", "FinanceRetriever"],
    )
    
    # Build graph
    research_graph = StateGraph(ResearchTeamState)
    research_graph.add_node("Search", search_node)
    research_graph.add_node("FinanceRetriever", research_node)
    research_graph.add_node("supervisor", supervisor_agent)
    
    research_graph.add_edge("Search", "supervisor")
    research_graph.add_edge("FinanceRetriever", "supervisor")
    research_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"Search": "Search", "FinanceRetriever": "FinanceRetriever", "FINISH": END},
    )
    
    research_graph.set_entry_point("supervisor")
    return research_graph.compile()

def enter_chain(message: str):
    """Entry point for research chain"""
    return {"messages": [HumanMessage(content=message)]}

def initialize_system(openai_key: str, tavily_key: str, cohere_key: str = None):
    """Initialize the entire RAG system"""
    global compiled_research_graph, rag_graph
    
    setup_environment(openai_key, tavily_key, cohere_key)
    
    # Load and process documents
    documents = load_and_chunk_documents()
    retriever = create_vectorstore(documents)
    
    # Create RAG graph
    rag_graph = create_rag_graph(retriever)
    
    # Create research graph
    compiled_research_graph = create_research_graph(retriever)

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="AIE7 Certification RAG API is running"
    )


@app.post("/test", response_model=PredictResponse)
async def test_endpoint(request: PredictRequest):
    """Test endpoint that works without API keys"""
    return PredictResponse(
        response=f"Test response received: {request.question}",
        context=["This is a test response that doesn't require API keys"]
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    x_openai_key: str = Header(None, alias="X-OpenAI-Key"),
    x_tavily_key: str = Header(None, alias="X-Tavily-Key"),
    x_cohere_key: str = Header(None, alias="X-Cohere-Key")
):
    """Main prediction endpoint"""
    try:
        # Initialize system with API keys from headers if not already initialized
        if compiled_research_graph is None:
            if not x_openai_key or not x_tavily_key:
                raise HTTPException(
                    status_code=400,
                    detail="OpenAI and Tavily API keys are required in headers"
                )
            
            try:
                initialize_system(x_openai_key, x_tavily_key, x_cohere_key)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize system: {str(e)}"
                )
        
        # Create research chain
        research_chain = enter_chain | compiled_research_graph
        
        # Process the question
        results = []
        print(f"DEBUG: Processing question: {request.question}")
        for step in research_chain.stream(
            request.question, 
            {"recursion_limit": Config.RECURSION_LIMIT}
        ):
            if "__end__" not in step:
                print(f"DEBUG: Step result: {step}")
                results.append(step)
        
        # Extract final response
        final_response = "I don't know."
        context_info = []
        
        if results:
            # Get the last meaningful response
            for result in reversed(results):
                for key, value in result.items():
                    if key != "supervisor" and "messages" in value:
                        messages = value["messages"]
                        if messages and hasattr(messages[0], 'content'):
                            final_response = messages[0].content
                            break
                if final_response != "I don't know.":
                    break
        
        return PredictResponse(
            response=final_response,
            context=context_info
        )
        
    except Exception as e:
        import traceback
        error_details = f"Prediction failed: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"ERROR: {error_details}")  # Log to console
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize system on startup - removed as we'll initialize on first request
# @app.on_event("startup")
# async def startup_event():
#     """Initialize system on FastAPI startup"""
#     # System will be initialized on first API call with keys

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

requirements = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.5.0",
    "langchain>=0.1.0",
    "langchain-community>=0.0.10",
    "langchain-openai>=0.0.5",
    "langchain-cohere>=0.1.0",
    "langgraph>=0.0.20",
    "tiktoken>=0.5.0",
    "qdrant-client>=1.7.0",
    "pymupdf>=1.23.0",
    "tavily-python>=0.3.0",
    "nest-asyncio>=1.5.0",
    "python-multipart>=0.0.6"
]