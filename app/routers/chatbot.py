from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
import asyncio
from typing import List, Any, Optional, Dict, Tuple
from pydantic import BaseModel
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.groq import Groq
from app.llama_index_server.index_storage import Models, ApiKeys
from app.data.messages.chat import ChatRequest, ChatResponse
from app.llama_index_server import agent_server
from app.llama_index_server.chat_message_dao import ChatMessageDao
from app.data.models.mongodb import Message
from app.utils.log_util import logger
from app.utils.data_consts import API_TIMEOUT

chatbot_router = APIRouter()
# Endpoint
@chatbot_router.post("/")
async def hello_world():
    return {"message": "Hello, World!"}

# Non-streaming chat
@chatbot_router.post(
    "/chat-non-streaming",
    response_model=ChatResponse,
    description="Chat with the ai bot in a non streaming way."
)
async def chat(request: ChatRequest)->ChatResponse:
    logger.info("Non streaming chat")
    conversation_id = request.conversation_id
    message = await asyncio.wait_for(agent_server.chat(request.content, conversation_id), timeout=API_TIMEOUT)
    return ChatResponse(data=message)

# Streaming chat
@chatbot_router.post(
    "/chat-streaming",
    description="Chat with the ai bot in a streaming way."
)
async def streaming_chat(request: ChatRequest):
    logger.info("Streaming chat")
    conversation_id = request.conversation_id
    return StreamingResponse(
        agent_server.stream_chat(request.content, conversation_id),
        media_type='text/plain'
    )

# Reset the context of the chat
@chatbot_router.post("/resetContext")
def reset_context(request: ChatRequest):
    agent = agent_server.get_chat_engine(request.conversation_id)
    agent.reset()
    return {"message": "Chat Context Resetted!"}

# Generate the title for the chat
@chatbot_router.post("/getChatTitle")
def get_chat_title(request: ChatRequest):
    summarization_prompt = f"""Summarize this query and write a suitable title from the following: "{request.content}. "
Only return the title in your response."""
    mixtral_groq = Groq(model=Models.GROQ_MIXTRAL, api_key=ApiKeys().get_api_key(org_name="groq"))
    response = mixtral_groq.complete(summarization_prompt)

    return {"title": response}