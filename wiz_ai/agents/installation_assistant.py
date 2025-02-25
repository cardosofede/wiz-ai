from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

from models.notion.knowledge_base import KnowledgeBaseDocument, NotionKnowledgeVectorDocument
from networks import EmbeddingModelSingleton
from wiz_ai.settings import settings

@dataclass
class ConversationContext:
    """State for the installation agent conversation."""
    current_message: str
    problem_descriptions: List[str] = field(default_factory=list)
    problem_statuses: List[str] = field(default_factory=list)
    retrieved_docs: List[KnowledgeBaseDocument] = field(default_factory=list)

@dataclass
class InstallationResponse:
    """Structured response from the installation agent."""
    response: str
    problem_descriptions: List[str]
    problem_statuses: List[str]

installation_agent = Agent(
    model=settings.GEMINI_MODEL_ID,
    system_prompt=(
        "You are a helpful assistant that helps Hummingbot users install and run the software. "
        "You have access to relevant documentation through the retrieve_docs tool. "
        "Follow these guidelines:\n"
        "1. First, use retrieve_docs to find relevant documentation for the user's query\n"
        "2. Analyze if this is a new problem or related to existing problems\n"
        "3. Provide clear, step-by-step instructions based on the documentation\n"
        "4. Update the problems list with new information\n"
        "5. If a problem is solved, mark its status as 'resolved'\n"
        "6. Always suggest next steps if the current solution doesn't work\n\n"
        "For each problem, provide two parallel lists:\n"
        "- problem_descriptions: List of problem descriptions\n"
        "- problem_statuses: List of corresponding statuses ('pending'/'in_progress'/'resolved')\n"
        "The lists must have the same length, where each index represents one problem."
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

def convert_to_dict_format(descriptions: List[str], statuses: List[str]) -> Dict[str, Dict[str, str]]:
    """Convert parallel lists to the dictionary format expected by the Discord bot."""
    return {
        f"problem_{i}": {"description": desc, "status": status}
        for i, (desc, status) in enumerate(zip(descriptions, statuses))
    }

async def process_message(message_content: str, problems_summary: Optional[Dict[str, Dict[str, str]]] = None) -> Tuple[str, Dict[str, Dict[str, str]]]:
    """Process a new message and generate a response using the installation agent."""
    # Convert existing problems from dict format to parallel lists
    current_descriptions = []
    current_statuses = []
    if problems_summary:
        for problem in problems_summary.values():
            current_descriptions.append(problem["description"])
            current_statuses.append(problem["status"])
    
    context = ConversationContext(
        current_message=message_content,
        problem_descriptions=current_descriptions,
        problem_statuses=current_statuses
    )
    
    # Format the input to include problems context
    problems_context = ""
    if current_descriptions:
        problems_list = "\n".join(
            f"- {desc} (Status: {status})"
            for desc, status in zip(current_descriptions, current_statuses)
        )
        problems_context = f"""Previous installation problems:
{problems_list}

"""
    
    input_text = f"""{problems_context}Current question:
{message_content}"""
    
    result = await installation_agent.run(
        input_text,
        deps=context,
        usage_limits=UsageLimits(request_limit=2, total_tokens_limit=2000)
    )
    
    # Convert the response back to the dictionary format
    updated_problems = convert_to_dict_format(
        result.data.problem_descriptions,
        result.data.problem_statuses
    )
    
    return result.data.response, updated_problems 