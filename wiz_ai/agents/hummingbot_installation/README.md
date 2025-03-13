# Hummingbot Installation Assistant

This LangGraph-based assistant helps users install Hummingbot by providing guidance on installation methods (Docker or Source), handling OS-specific instructions, and retrieving relevant documentation.

## Features

- **Semantic Document Retrieval**: Searches a vector store for relevant documentation based on user queries
- **Conversation Analysis**: Detects installation method, operating system, and tracks conversation progress
- **Conditional Responses**: Tailors responses based on the conversation state
- **Support Escalation**: Automatically escalates to support team after 9 iterations if not resolved
- **Context Maintenance**: Maintains conversation history and summary

## Installation Methods

The assistant helps with two main installation methods:

1. **Docker Installation**:
   - The simplest method requiring only Docker installed on the system
   - Works on any OS that supports Docker (Windows, macOS, Linux)

2. **Source Installation**:
   - Requires Anaconda installed on the system
   - Windows users need WSL (Windows Subsystem for Linux) installed first
   - MacOS and Linux users need Anaconda

## How it Works

The assistant uses a LangGraph workflow with the following nodes:

1. `retrieve_documents`: Retrieves relevant documentation from vector store
2. `analyze_conversation`: Analyzes the conversation to detect installation method, OS, etc.
3. `generate_response`: Generates a response based on the current state

The graph has conditional edges:
- If the conversation is solved, it ends
- If the conversation requires escalation, it tags support team
- Otherwise, it continues the conversation flow

## Usage

```python
import asyncio
from wiz_ai.agents.hummingbot-installation.studio.installation_assistant import process_message

# Process a new message
async def example():
    # For a new conversation
    state = await process_message("How do I install Hummingbot on Windows?")
    
    # Continue the conversation with existing state
    updated_state = await process_message("I'm using Docker", state)
    
    # Get the last AI response
    ai_response = next((m["content"] for m in reversed(updated_state["messages"]) 
                     if m["role"] == "ai"), "")
    print(ai_response)

asyncio.run(example())
```

## Demo Application

The module includes a demo application that can be run directly:

```bash
python -m wiz_ai.agents.hummingbot-installation.studio.installation_assistant
```

## Integration with Other Systems

The installation assistant can be integrated with chat platforms like Discord by:

1. Maintaining conversation state in a database
2. Processing incoming messages through the `process_message` function
3. Displaying the resulting AI messages to users 