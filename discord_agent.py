from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional

import discord
from discord.ext import commands
from discord import ButtonStyle
from discord.ui import View, Button
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import logfire

logfire.configure()
load_dotenv()

from wiz_ai.settings import settings
from wiz_ai.agents.installation_assistant import process_message


# -------------------- Environment Variables --------------------
# Main bot credentials and channels
MAIN_CHANNEL_IDS = [int(cid) for cid in settings.MAIN_CHANNEL_IDS.split(',') if cid.strip()]

# -------------------- Main Bot Setup --------------------
intents = discord.Intents.default()
intents.message_content = True  # Needed for on_message
bot = commands.Bot(command_prefix="!", intents=intents)

class State(Enum):
    START = 0
    IN_PROGRESS = 1
    END = 2

class Attachment(BaseModel):
    filename: str
    url: str
    content_type: Optional[str]
    size: int

class Message(BaseModel):
    content: str
    author_id: int
    is_bot: bool
    timestamp: datetime
    attachments: List[Attachment] = []

class Conversation(BaseModel):
    messages: List[Message]
    user_id: int
    state: State
    channel_id: int
    last_activity: datetime = Field(default_factory=datetime.now)
    problems_summary: str

    def add_message(self, content: str, author_id: int, is_bot: bool, attachments: List[discord.Attachment] = None):
        message_attachments = []
        if attachments:
            for att in attachments:
                message_attachments.append(
                    Attachment(
                        filename=att.filename,
                        url=att.url,
                        content_type=att.content_type,
                        size=att.size
                    )
                )

        self.messages.append(
            Message(
                content=content,
                author_id=author_id,
                is_bot=is_bot,
                timestamp=datetime.now(),
                attachments=message_attachments
            )
        )
        self.last_activity = datetime.now()

class ConversationManager:
    def __init__(self):
        self.channel_conversations: Dict[int, List[Conversation]] = {}

    def get_active_conversation(self, channel_id: int, user_id: int) -> Optional[Conversation]:
        if channel_id not in self.channel_conversations:
            return None
        return next(
            (conv for conv in self.channel_conversations[channel_id]
             if conv.user_id == user_id and conv.state != State.END),
            None
        )

    def create_conversation(self, channel_id: int, user_id: int) -> Conversation:
        if channel_id not in self.channel_conversations:
            self.channel_conversations[channel_id] = []

        conv = Conversation(
            messages=[],
            user_id=user_id,
            state=State.START,
            channel_id=channel_id
        )
        self.channel_conversations[channel_id].append(conv)
        return conv

    def add_message(self, channel_id: int, user_id: int, content: str, is_bot: bool, attachments: List[discord.Attachment] = None) -> Conversation:
        conv = self.get_active_conversation(channel_id, user_id)
        if not conv:
            conv = self.create_conversation(channel_id, user_id)

        conv.add_message(content, user_id, is_bot, attachments)
        if not is_bot and conv.state == State.START:
            conv.state = State.IN_PROGRESS
        return conv

class SolveButton(Button):
    def __init__(self, conversation: Conversation):
        super().__init__(
            style=ButtonStyle.success,
            label="Mark as Solved",
            custom_id=f"solve_{conversation.user_id}"
        )
        self.conversation = conversation

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id == self.conversation.user_id:
            self.conversation.state = State.END
            await interaction.response.send_message("Conversation marked as solved! Thank you for using our bot.", ephemeral=True)
            self.view.stop()
        else:
            await interaction.response.send_message("Only the conversation starter can mark it as solved.", ephemeral=True)

class ConversationView(View):
    def __init__(self, conversation: Conversation):
        super().__init__(timeout=None)
        self.add_item(SolveButton(conversation))

# Initialize the conversation manager
conversation_manager = ConversationManager()

# -------------------- on_ready and on_message for Main Bot --------------------
@bot.event
async def on_ready():
    try:
        await bot.tree.sync()
        logfire.info("Slash commands synced successfully.")
    except Exception as e:
        logfire.error(f"Error syncing commands: {e}")
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.channel.id in MAIN_CHANNEL_IDS:
        # Get or create conversation
        conv = conversation_manager.get_active_conversation(
            channel_id=message.channel.id,
            user_id=message.author.id
        )
        if not conv:
            conv = conversation_manager.create_conversation(
                channel_id=message.channel.id,
                user_id=message.author.id
            )

        # Add user message to conversation
        conv.add_message(
            content=message.content,
            author_id=message.author.id,
            is_bot=False,
            attachments=message.attachments
        )
        
        # Process message and get structured response
        response_text, updated_problems_summary = await process_message(
            message_content=message.content,
            previous_summary=conv.problems_summary
        )
        
        conv.problems_summary = updated_problems_summary
        # Add bot response to conversation
        conv.add_message(
            content=response_text,
            author_id=message.author.id,
            is_bot=True,
            attachments=[]
        )

        # Create response embed with problem summary if available
        embed = discord.Embed(title="Useful links", color=discord.Color.blue())
        embed.add_field(
            name="Documentation",
            value="[Installation via Docker](https://hummingbot.org/installation/docker/)\n"
                  "[Installation via Source](https://hummingbot.io/installation/source/)",
            inline=False
        )
        embed.set_footer(text="Use the button below to mark the conversation as solved.")

        # Send response with conversation view and embed
        view = ConversationView(conv)
        await message.channel.send(response_text, view=view, embed=embed)

# -------------------- Start the Main Bot --------------------
async def main():
    try:
        async with bot:
            await bot.start(settings.MAIN_DISCORD_TOKEN)
    except discord.LoginFailure:
        logfire.error("Failed to log in: Invalid token")
    except discord.ConnectionClosed:
        logfire.error("Connection closed unexpectedly")
    except Exception as e:
        logfire.error(f"An error occurred: {e}")
    finally:
        if not bot.is_closed():
            await bot.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
