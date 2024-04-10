from typing import Union
from llama_index.core import Prompt
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.utils.log_util import logger
from app.utils import data_util
from app.llama_index_server.index_storage import IndexStorage
from app.llama_index_server.chat_message_dao import ChatMessageDao
from app.llama_index_server.hpc_query_engine_tool import build_query_engine_tool
from app.llama_index_server.code_assist_tool import build_code_assist_tool
from app.data.models.mongodb import Message

executor = ThreadPoolExecutor(max_workers=100)

IRRELEVANT_QUESTION = {
    "default_answer_id": "irrelevant_question",
    "default_answer": "This question is not relevant to golf, please ask a question related to golf.",
}

def get_default_answer_id():
    return IRRELEVANT_QUESTION["default_answer_id"]

def get_default_answer():
    return IRRELEVANT_QUESTION["default_answer"]

CONTEXT = """\
You are a hpc helpful assistant who is an expert on Ho Chi Minh University of Technology's HPC Lab's information.\
    You will answer questions about HPC Lab as in the persona of a friendly assistant \
    and a top hpc expert. \
    If there is a code block or multiple code blocks in the question, extract the code block and pass to the 'code_assistant_tool'
"""
PROMPT_TEMPLATE_FOR_QUERY_ENGINE = (
    "Assume you are the administrator of HPC Lab system glad to answer questions from HPC Lab's users, "
    "if the question has anything to do with HPC Lab, or the Supernode-XP system, or HPC knowledge, "
    "please give short, simple, accurate, precise answer to the question, "
    "limited to 2000 words maximum. If the question has nothing to do with HPC Lab at all, please answer "
    f"'{get_default_answer_id()}'.\n"
    "The question is: {query_str}\n"
)
SYSTEM_PROMPT_TEMPLATE_FOR_CHAT_ENGINE = (
    "Your are an expert Q&A system that can find relevant information using the tools at your disposal, and you have "
    "great knowledge about HPC Lab.\n"
    "The tools can access a set of typical questions a member of HPC Lab might ask.\n"
    "If the user's query matches one of those typical questions, stop and return the matched question immediately.\n"
    "If the user's query doesn't match any of those typical questions, "
    "please give short, simple, accurate, precise answer to the question, limited to 2000 words maximum.\n"
    "You may need to combine the chat history to fully understand the query of the user.\n"
)

def get_default_answer_id():
    return IRRELEVANT_QUESTION["default_answer_id"]

def cleanup_for_test():
    return IndexStorage().mongo().cleanup_for_test()

chat_message_dao = ChatMessageDao()
def get_chat_engine(conversation_id: str):
    local_query_engine_tool = build_query_engine_tool()
    code_assist_tool = build_code_assist_tool()
    chat_tools = [local_query_engine_tool, code_assist_tool]
    chat_llm = Ollama(model="ontocord/vistral:q4_0", request_timeout=60.0)
    chat_history = chat_message_dao.get_chat_history(conversation_id)
    chat_history = [ChatMessage(role=c.role, content=c.content) for c in chat_history]
    return ReActAgent.from_tools(
        tools=chat_tools,
        llm=chat_llm,
        chat_history=chat_history,
        verbose=True,
        system_prompt=SYSTEM_PROMPT_TEMPLATE_FOR_CHAT_ENGINE,
        context=CONTEXT,
    )

def get_response_text_from_chat(agent_chat_response):
    return agent_chat_response.response

# Non-streaming Chat
async def chat(query_text: str, conversation_id: str) -> Message:
    # we will not index chat messages in vector store, but will save them in mongodb
    data_util.assert_not_none(query_text, "query content cannot be none")
    user_message = ChatMessage(role=MessageRole.USER, content=query_text)
    # save immediately, since the following steps may take a while and throw exceptions
    chat_message_dao.save_chat_history(conversation_id, user_message)
    chat_engine = get_chat_engine(conversation_id)

    loop = asyncio.get_running_loop()
    
    agent_chat_response = await loop.run_in_executor(executor, chat_engine.chat, query_text)
    response_text = get_response_text_from_chat(agent_chat_response)
    response_text = get_default_answer() if get_default_answer_id() in response_text else response_text

    bot_message = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
    chat_message_dao.save_chat_history(conversation_id, bot_message)

    return Message.from_chat_message(conversation_id, bot_message)

# Streaming Chat
async def stream_chat(query_text: str, conversation_id: str):
    # we will not index chat messages in vector store, but will save them in mongodb
    data_util.assert_not_none(query_text, "query content cannot be none")
    user_message = ChatMessage(role=MessageRole.USER, content=query_text)
    # save immediately, since the following steps may take a while and throw exceptions
    chat_message_dao.save_chat_history(conversation_id, user_message)
    chat_engine = get_chat_engine(conversation_id)
    
    agent_chat_response = await chat_engine.astream_chat(query_text)
    async for token in agent_chat_response.async_response_gen():
        # if await request.is_disconnected():
        #     break
        yield token
    response_text = get_response_text_from_chat(agent_chat_response)

    bot_message = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
    chat_message_dao.save_chat_history(conversation_id, bot_message)