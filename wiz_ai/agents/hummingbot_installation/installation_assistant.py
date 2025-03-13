from dataclasses import field
from enum import Enum
from typing import Dict, List, Any

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langgraph.graph import StateGraph, START, END, MessagesState

from wiz_ai.settings import settings

load_dotenv()

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(
    host=settings.QDRANT_DATABASE_HOST,
    port=settings.QDRANT_DATABASE_PORT,
)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="knowledge_base",
    embedding=embeddings,
)

# Define our models/enums
class ConversationState(str, Enum):
    ACTIVE = "active"
    SOLVED = "solved"
    ESCALATED = "escalated"

class InstallationMethod(str, Enum):
    DOCKER = "docker"
    SOURCE = "source"
    UNKNOWN = "unknown"

class UserOS(str, Enum):
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"


# The state for our installation assistant graph
class State(MessagesState):
    """State for the installation assistant conversation"""
    summary: str = ""
    retrieved_docs: List[Dict] = Field(default_factory=list)
    iteration_count: int = 0
    conversation_state: ConversationState = ConversationState.ACTIVE
    detected_installation_method: InstallationMethod = InstallationMethod.UNKNOWN
    detected_os: UserOS = UserOS.UNKNOWN
    needs_support_escalation: bool = False
    
# Model for parsing the assistant's analysis
class InstallationAssistantAnalysis(BaseModel):
    """Structure for the assistant's analysis of the conversation"""
    updated_summary: str = Field(description="Updated summary of the problem and progress")
    is_solved: bool = Field(description="Whether the problem has been solved")
    detected_installation_method: InstallationMethod = Field(description="The detected installation method")
    detected_os: UserOS = Field(description="The detected operating system") 
    needs_support_escalation: bool = Field(description="Whether to escalate to support team")

# Initialize the LLM
llm = ChatOpenAI(model=settings.OPENAI_MODEL_ID, temperature=0.3)


def retrieve_documents(state: State) -> Dict:
    """Retrieve relevant documents based on the latest user message"""
    
    # Get the most recent user message
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_messages:
        return {}
    
    latest_message = user_messages[-1].content
    
    # Check for previously retrieved docs to avoid duplicates
    already_retrieved_doc_ids = {doc.get("id") for doc in state.get("retrieved_docs", [])}
    
    # Use langchain's vector store to search for similar documents
    docs = vector_store.similarity_search(latest_message, k=3)
    
    # Filter out already retrieved documents and format them
    new_docs = []
    for doc in docs:
        doc_id = doc.metadata.get("document_id", str(hash(doc.page_content)))
        if doc_id not in already_retrieved_doc_ids:
            new_docs.append({
                "id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata
            })
    
    # Return new docs to add to the state
    if new_docs:
        # Create a new combined list - we're not using Annotated with operator.add anymore
        all_docs = list(state.get("retrieved_docs", [])) + new_docs
        return {
            "retrieved_docs": all_docs
        }
    return {}

def analyze_conversation(state: State) -> Dict:
    """Analyze the conversation to determine status, installation method, OS, etc."""
    
    # Create a JSON output parser
    parser = JsonOutputParser(pydantic_object=InstallationAssistantAnalysis)
    
    # Format messages for analysis
    messages_text = ""
    for i, msg in enumerate(state["messages"]):
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        messages_text += f"{role}: {msg.content}\n\n"
    
    # Create analysis prompt
    prompt = f"""Analyze the following conversation about Hummingbot installation.

CONVERSATION HISTORY:
{messages_text}

CURRENT SUMMARY:
{state["summary"] if state.get("summary") else "No summary available."}

Based on the conversation, provide:
1. An updated summary of the problem and progress
2. Whether the problem has been solved (user explicitly stated or clearly implied)
3. The detected installation method (docker, source, or unknown)
4. The detected operating system (windows, macos, linux, or unknown)
5. Whether the issue needs to be escalated to the support team

Installation methods information:
- docker: User is trying to install via Docker containers
- source: User is trying to install from source code using Anaconda

OS specific information:
- For Windows source installation: WSL (Windows Subsystem for Linux) is required
- For macOS source installation: Anaconda is required
- For Linux source installation: Anaconda is required
- For Docker installation: Docker must be installed regardless of OS

{parser.get_format_instructions()}
"""
    
    # Get structured analysis
    analysis_chain = llm.with_structured_output(InstallationAssistantAnalysis)
    result = analysis_chain.invoke(prompt)
    
    # Update conversation state based on analysis
    conversation_state = ConversationState.SOLVED if result.is_solved else ConversationState.ACTIVE
    
    # Check if we need to escalate to support (either from analysis or iteration count)
    needs_escalation = (
        result.needs_support_escalation or 
        (state["iteration_count"] >= 9 and conversation_state == ConversationState.ACTIVE)
    )
    
    if needs_escalation:
        conversation_state = ConversationState.ESCALATED
    
    return {
        "summary": result.updated_summary,
        "conversation_state": conversation_state,
        "detected_installation_method": result.detected_installation_method,
        "detected_os": result.detected_os,
        "needs_support_escalation": needs_escalation
    }

def generate_response(state: State) -> Dict:
    """Generate a response to the user based on the current state"""
    
    # Get the current documents and conversation history
    docs = state["retrieved_docs"]
    doc_content = "\n\n".join([f"DOCUMENT {i+1}:\n{doc['content']}" for i, doc in enumerate(docs)])
    
    # Format messages for context
    messages_text = ""
    for i, msg in enumerate(state["messages"][-5:]):  # Only include last 5 messages for context
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        messages_text += f"{role}: {msg.content}\n\n"
    
    # Create prompt based on conversation state
    if state["conversation_state"] == ConversationState.ESCALATED:
        system_prompt = f"""You are a helpful assistant that helps Hummingbot users install and run the software.
This conversation has been going on for a while without resolution. Tag a support team member to help.

CONVERSATION SUMMARY:
{state["summary"]}

RECENT CONVERSATION:
{messages_text}

RETRIEVED DOCUMENTS:
{doc_content}

INSTALLATION METHOD: {state["detected_installation_method"]}
OPERATING SYSTEM: {state["detected_os"]}

Explain to the user that you'll be tagging a support team member to help further, then tag @support-team. Be empathetic and summarize the current status and what's been tried so far.
"""
    elif state["conversation_state"] == ConversationState.SOLVED:
        system_prompt = f"""You are a helpful assistant that helps Hummingbot users install and run the software.
This conversation has been marked as SOLVED.

CONVERSATION SUMMARY:
{state["summary"]}

RECENT CONVERSATION:
{messages_text}

Express positivity about resolving the issue, provide a brief summary of what was done, and offer any final tips or resources that might be helpful for the user going forward.
"""
    else:
        system_prompt = f"""You are a helpful assistant that helps Hummingbot users install and run the software.
Always provide the specific commands (like `docker compose up -d`) in fenced code blocks if mentioned in the retrieved docs.

CONVERSATION SUMMARY:
{state["summary"]}

RECENT CONVERSATION:
{messages_text}

RETRIEVED DOCUMENTS:
{doc_content}

INSTALLATION METHOD: {state["detected_installation_method"]}
OPERATING SYSTEM: {state["detected_os"]}

Answer the user's question based on the retrieved documents. If the documents don't provide enough information, give your best guidance based on:
1. For source installation: Emphasize Anaconda requirement, and WSL for Windows users
2. For Docker installation: Emphasize Docker requirement and provide commands if available
3. Be clear, step-by-step, and suggest next steps the user should take
"""
    
    # Generate response
    response = llm.invoke([SystemMessage(content=system_prompt)])
    
    # Add the new message to the state
    return {
        "messages": [{"role": "ai", "content": response.content}],
        "iteration_count": state["iteration_count"] + 1
    }

def should_continue(state: State) -> str:
    """Determine if the conversation should continue or end"""
    if state["conversation_state"] == ConversationState.SOLVED or state["conversation_state"] == ConversationState.ESCALATED:
        return "end"
    return "continue"

    
# Create the graph builder with our State type
builder = StateGraph(State)

# Add nodes
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("analyze_conversation", analyze_conversation)
builder.add_node("generate_response", generate_response)

# Add conditional edges
builder.add_edge(START, "retrieve_documents")

# Connect the nodes
builder.add_edge("retrieve_documents", "analyze_conversation")
builder.add_edge("analyze_conversation", "generate_response")

# Add conditional edges for ending or continuing the conversation
builder.add_conditional_edges(
    "generate_response",
    {
        END: lambda state: state["conversation_state"] in [ConversationState.SOLVED, ConversationState.ESCALATED],
        "retrieve_documents": lambda state: state["conversation_state"] == ConversationState.ACTIVE,
    }
)

# Compile the graph
graph = builder.compile()

# Example of how to properly invoke the graph with initial state
initial_state = {"messages": [{"role": "human", "content": "I want to install hummingbot on my Mac"}],
                 "iteration_count": 0}
result = graph.invoke(initial_state)

# Add metadata to the graph for UI registration
graph.name = "Installation Assistant"
graph.description = "A graph for helping users with Hummingbot installation issues."
