import os
import functools
import operator
import logging
import time
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

# Logging Configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

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

# Request/Response Logging Middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Log incoming request
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.debug(f"Request headers: {dict(request.headers)}")
    
    # Log request body for POST requests (be careful with sensitive data)
    if request.method == "POST":
        try:
            # Read the body
            body = await request.body()
            if body:
                # Try to decode as JSON for logging
                try:
                    import json
                    body_json = json.loads(body.decode())
                    logger.debug(f"Request body: {body_json}")
                except:
                    logger.debug(f"Request body (raw): {body[:500]}...")  # Limit to first 500 chars
            
            # Create a new request with the same body for the endpoint
            from fastapi import Request
            from starlette.requests import Request as StarletteRequest
            
            async def receive():
                return {"type": "http.request", "body": body}
            
            # Recreate request with body
            request = Request(request.scope, receive)
            
        except Exception as e:
            logger.warning(f"Could not log request body: {e}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(f"Response: {response.status_code} - Processing time: {process_time:.4f}s")
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

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
    logger.info("Setting up environment variables")
    logger.debug(f"OpenAI key provided: {'Yes' if openai_key else 'No'}")
    logger.debug(f"Tavily key provided: {'Yes' if tavily_key else 'No'}")
    logger.debug(f"Cohere key provided: {'Yes' if cohere_key else 'No'}")
    
    # For debugging with test keys, disable tracing to avoid auth errors
    if openai_key == "test-key":
        logger.warning("Using test keys - disabling LangSmith tracing")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        logger.debug("Setting up LangSmith tracing")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    if cohere_key:
        os.environ["COHERE_API_KEY"] = cohere_key
    else:
        logger.warning("No Cohere key provided, using default")
        os.environ["COHERE_API_KEY"] = "default-cohere-key"  # Provide default
    
    project_name = f"AIE7-cert-{uuid4().hex[:8]}"
    os.environ.setdefault("LANGCHAIN_PROJECT", project_name)
    logger.info(f"LangChain project set to: {project_name}")
    
    nest_asyncio.apply()
    logger.debug("Nest asyncio applied")

def tiktoken_len(text: str) -> int:
    """Calculate token length using tiktoken"""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)

def load_and_chunk_documents():
    """Load and chunk PDF documents from data directory"""
    logger.info(f"Loading documents from: {Config.DATA_DIR}")
    
    try:
        directory_loader = DirectoryLoader(
            Config.DATA_DIR, 
            glob=Config.PDF_GLOB, 
            loader_cls=PyMuPDFLoader
        )
        finance_resources = directory_loader.load()
        logger.info(f"Loaded {len(finance_resources)} documents")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=tiktoken_len,
        )
        
        chunks = text_splitter.split_documents(finance_resources)
        logger.info(f"Split documents into {len(chunks)} chunks")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}", exc_info=True)
        raise

def create_vectorstore(documents):
    """Create Qdrant vectorstore from documents"""
    logger.info(f"Creating vectorstore with {len(documents)} documents")
    
    try:
        embedding_model = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        logger.debug(f"Using embedding model: {Config.EMBEDDING_MODEL}")
        
        qdrant_vectorstore = Qdrant.from_documents(
            documents=documents,
            embedding=embedding_model,
            location=":memory:"
        )
        logger.info("Vectorstore created successfully")
        
        retriever = qdrant_vectorstore.as_retriever()
        logger.debug("Retriever created from vectorstore")
        
        return retriever
        
    except Exception as e:
        logger.error(f"Error creating vectorstore: {str(e)}", exc_info=True)
        raise

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
        logger.debug(f"retrieve_information called with query: {query}")
        start_time = time.time()
        
        try:
            result = contextual_compression_retrieval_chain.invoke({"question": query})
            processing_time = time.time() - start_time
            logger.debug(f"retrieve_information completed in {processing_time:.4f}s")
            logger.debug(f"retrieve_information result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in retrieve_information: {str(e)}", exc_info=True)
            raise
    
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
    
    logger.info("Starting system initialization")
    start_time = time.time()
    
    try:
        setup_environment(openai_key, tavily_key, cohere_key)
        
        # Load and process documents
        logger.info("Loading and processing documents")
        doc_start = time.time()
        documents = load_and_chunk_documents()
        logger.info(f"Document processing completed in {time.time() - doc_start:.2f}s")
        
        # Create vectorstore
        logger.info("Creating vectorstore")
        vector_start = time.time()
        retriever = create_vectorstore(documents)
        logger.info(f"Vectorstore creation completed in {time.time() - vector_start:.2f}s")
        
        # Create RAG graph
        logger.info("Creating RAG graph")
        rag_start = time.time()
        rag_graph = create_rag_graph(retriever)
        logger.info(f"RAG graph creation completed in {time.time() - rag_start:.2f}s")
        
        # Create research graph
        logger.info("Creating research graph")
        research_start = time.time()
        compiled_research_graph = create_research_graph(retriever)
        logger.info(f"Research graph creation completed in {time.time() - research_start:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"System initialization completed successfully in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}", exc_info=True)
        raise

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    logger.debug("Health check requested")
    
    system_status = "healthy" if compiled_research_graph is not None else "not_initialized"
    message = "AIE7 Certification RAG API is running"
    
    if system_status == "not_initialized":
        message += " (System not yet initialized - awaiting first prediction request)"
    
    logger.debug(f"Health check response: {system_status}")
    
    return HealthResponse(
        status=system_status,
        message=message
    )


@app.post("/test", response_model=PredictResponse)
async def test_endpoint(request: PredictRequest):
    """Test endpoint that works without API keys"""
    logger.info(f"Test endpoint called with question: {request.question}")
    
    response = PredictResponse(
        response=f"Test response received: {request.question}",
        context=["This is a test response that doesn't require API keys"]
    )
    
    logger.debug(f"Test endpoint returning: {response}")
    return response


@app.post("/debug", response_model=PredictResponse)
async def debug_endpoint(request: PredictRequest):
    """Debug endpoint to test system state"""
    logger.info(f"Debug endpoint called with question: {request.question}")
    
    try:
        # Check environment variables
        openai_key = os.environ.get("OPENAI_API_KEY", "Not set")
        tavily_key = os.environ.get("TAVILY_API_KEY", "Not set") 
        cohere_key = os.environ.get("COHERE_API_KEY", "Not set")
        
        logger.debug(f"Environment - OpenAI: {'Set' if openai_key != 'Not set' else 'Not set'}")
        logger.debug(f"Environment - Tavily: {'Set' if tavily_key != 'Not set' else 'Not set'}")
        logger.debug(f"Environment - Cohere: {'Set' if cohere_key != 'Not set' else 'Not set'}")
        
        # Check global variables
        logger.debug(f"Global state - compiled_research_graph: {'initialized' if compiled_research_graph else 'None'}")
        logger.debug(f"Global state - rag_graph: {'initialized' if rag_graph else 'None'}")
        
        # Check if data directory exists
        import os
        data_dir_exists = os.path.exists(Config.DATA_DIR)
        logger.debug(f"Data directory '{Config.DATA_DIR}' exists: {data_dir_exists}")
        
        if data_dir_exists:
            pdf_files = []
            for _, _, files in os.walk(Config.DATA_DIR):
                pdf_files.extend([f for f in files if f.endswith('.pdf')])
            logger.debug(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        
        response_text = f"Debug info logged. System state: compiled_research_graph={'initialized' if compiled_research_graph else 'None'}, data_dir_exists={data_dir_exists}"
        
        response = PredictResponse(
            response=response_text,
            context=[f"Debug response for: {request.question}"]
        )
        
        logger.debug(f"Debug endpoint returning: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Debug endpoint failed: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    x_openai_key: str = Header(None, alias="X-OpenAI-Key"),
    x_tavily_key: str = Header(None, alias="X-Tavily-Key"),
    x_cohere_key: str = Header(None, alias="X-Cohere-Key")
):
    """Main prediction endpoint"""
    logger.info(f"Prediction request received for question: {request.question}")
    prediction_start = time.time()
    
    try:
        # Initialize system with API keys from headers if not already initialized
        if compiled_research_graph is None:
            logger.info("System not initialized, checking API keys")
            if not x_openai_key or not x_tavily_key:
                logger.error("Missing required API keys")
                raise HTTPException(
                    status_code=400,
                    detail="OpenAI and Tavily API keys are required in headers"
                )
            
            try:
                logger.info("Initializing system with provided API keys")
                initialize_system(x_openai_key, x_tavily_key, x_cohere_key)
            except Exception as e:
                logger.error(f"System initialization failed: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize system: {str(e)}"
                )
        else:
            logger.debug("System already initialized, proceeding with prediction")
        
        # Create research chain
        logger.debug("Creating research chain")
        research_chain = enter_chain | compiled_research_graph
        
        # Process the question
        results = []
        logger.info(f"Processing question through research chain: {request.question}")
        processing_start = time.time()
        
        for step in research_chain.stream(
            request.question, 
            {"recursion_limit": Config.RECURSION_LIMIT}
        ):
            if "__end__" not in step:
                logger.debug(f"Research chain step: {step}")
                results.append(step)
        
        processing_time = time.time() - processing_start
        logger.info(f"Research chain processing completed in {processing_time:.4f}s")
        
        # Extract final response
        final_response = "I don't know."
        context_info = []
        
        if results:
            logger.debug("Extracting final response from results")
            # Get the last meaningful response
            for result in reversed(results):
                for key, value in result.items():
                    if key != "supervisor" and "messages" in value:
                        messages = value["messages"]
                        if messages and hasattr(messages[0], 'content'):
                            final_response = messages[0].content
                            logger.debug(f"Final response extracted: {final_response}")
                            break
                if final_response != "I don't know.":
                    break
        else:
            logger.warning("No results returned from research chain")
        
        total_time = time.time() - prediction_start
        logger.info(f"Prediction completed in {total_time:.4f}s")
        
        return PredictResponse(
            response=final_response,
            context=context_info
        )
        
    except HTTPException as he:
        # Log HTTP exceptions before re-raising
        logger.error(f"HTTP Exception in predict: {he.status_code} - {he.detail}")
        raise
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_message = str(e)
        full_traceback = traceback.format_exc()
        
        # Log detailed error information
        logger.error(f"Prediction failed with {error_type}: {error_message}")
        logger.error(f"Full traceback:\n{full_traceback}")
        
        # Also log the state of global variables for debugging
        logger.error(f"System state - compiled_research_graph: {'initialized' if compiled_research_graph else 'None'}")
        logger.error(f"System state - rag_graph: {'initialized' if rag_graph else 'None'}")
        
        # Return detailed error message
        detailed_error = f"{error_type}: {error_message}"
        raise HTTPException(status_code=500, detail=f"Prediction failed: {detailed_error}")

# Initialize system on startup - removed as we'll initialize on first request
# @app.on_event("startup")
# async def startup_event():
#     """Initialize system on FastAPI startup"""
#     # System will be initialized on first API call with keys

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    logger.info("Debug logging enabled - check app.log for detailed logs")
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