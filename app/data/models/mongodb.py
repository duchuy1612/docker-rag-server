from pydantic import Field, BaseModel
from typing import Optional
from llama_index.core.llms import ChatMessage, MessageRole
from app.utils import data_util

class CollectionModel(BaseModel):
    """
        Base class for all the collections stored in MongoDB.
    """
    @staticmethod
    def db_name():
        return "ai_bot"

    @staticmethod
    def collection_name():
        return None
    
class Message(CollectionModel):
    conversation_id: str = Field(..., description="Unique id of the conversation")
    role: MessageRole = Field(..., description="Author of the chat message")
    content: str = Field(..., description="Content of the chat message")
    timestamp: int = Field(..., description="Time when this chat message was sent, in milliseconds")
    time: Optional[str] = Field(None, description="Time when this chat message was sent, in human readable format")

    @staticmethod
    def collection_name():
        return "chat_message"

    @staticmethod
    def from_chat_message(conversation_id: str, chat_message: ChatMessage):
        return Message(
            conversation_id=conversation_id,
            role=chat_message.role,
            content=chat_message.content,
            timestamp=data_util.get_current_milliseconds(),
        )

    def __init__(self, **data):
        super().__init__(**data)
        self.time = data_util.milliseconds_to_human_readable(self.timestamp)