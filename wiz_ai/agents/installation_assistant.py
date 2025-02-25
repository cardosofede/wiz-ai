from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

from wiz_ai.models.notion.knowledge_base import KnowledgeBaseDocument, NotionKnowledgeVectorDocument
from wiz_ai.networks import EmbeddingModelSingleton
from wiz_ai.settings import settings

@dataclass
class ConversationContext:
    """State for the installation agent conversation."""
    current_message: str
    conversation_summary: str = ""  # Summary of the conversation so far
    retrieved_docs: List[KnowledgeBaseDocument] = field(default_factory=list)

@dataclass
class InstallationResponse:
    """Structured response from the installation agent."""
    response: str
    conversation_summary: str  # Updated summary including the current interaction

installation_agent = Agent(
    model=settings.GEMINI_MODEL_ID,
    system_prompt=(
        "You are a helpful assistant that helps Hummingbot users install and run the software. "
        "Always provide the specific commands (like `docker compose up -d`) in fenced code blocks if mentioned in the retrieved docs. "
        "You have access to relevant documentation through the retrieve_docs tool. "
        "Follow these guidelines:\n"
        "1. First, use retrieve_docs to find relevant documentation for the user's query\n"
        "2. Provide clear, step-by-step instructions based on the documentation\n"
        "3. Keep track of the conversation by maintaining a brief summary\n"
        "4. Always suggest next steps if needed"
    ),
    result_type=InstallationResponse,
    deps_type=ConversationContext
)

@installation_agent.tool
async def retrieve_docs(ctx: RunContext[ConversationContext], query: str) -> List[str]:
    """Search for relevant documentation based on the user's query."""
    embedding_model = EmbeddingModelSingleton()
    query_embedding = embedding_model(query, to_list=True)
    docs = NotionKnowledgeVectorDocument.search(query_embedding, limit=3)
    ctx.retrieved_docs = docs
    return [doc.content for doc in docs]

async def process_message(message_content: str, previous_summary: Optional[str] = None) -> Tuple[str, str]:
    """Process a new message and generate a response using the installation agent."""
    context = ConversationContext(
        current_message=message_content,
        conversation_summary=previous_summary or ""
    )
    
    # Format input with previous summary if it exists
    input_text = message_content
    if previous_summary:
        input_text = f"""Previous conversation summary:
{previous_summary}

Current question:
{message_content}"""
    
    result = await installation_agent.run(
        input_text,
        deps=context,
        usage_limits=UsageLimits(request_limit=2, total_tokens_limit=2000)
    )
    
    return result.data.response, result.data.conversation_summary 