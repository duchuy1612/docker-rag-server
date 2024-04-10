from pydantic import BaseModel, Field
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool, ToolMetadata

DEFAULT_NAME = "code_assist_function_tool"
DEFAULT_DESCRIPTION = (
    "useful for when you want to answer queries that have"
    " codes in them or the queries are code-related"
)

class CodeRetrievalModel(BaseModel):
    query: str = Field(..., description="natural language query string")

def code(query: str):
    """Use coder model to answer code-related questions"""
    deepseek_coder_model = Ollama(model="deepseek-coder:6.7b", request_timeout=60.0)
    return deepseek_coder_model.complete(query)

def build_code_assist_tool()->FunctionTool:
    code_assistant_tool = FunctionTool(
        fn=code,
        metadata=ToolMetadata(
            name=DEFAULT_NAME,
            description=DEFAULT_DESCRIPTION,
            fn_schema=CodeRetrievalModel,
        ),  
    )

    return code_assistant_tool