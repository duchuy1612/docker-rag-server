from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.data.messages.status_code import StatusCode
import asyncio
from app.routers.chatbot import chatbot_router
from app.utils.log_util import logger
import uvicorn
import time
import os

app = FastAPI(
    title="Api Definitions",
    servers=[
        {
            "url": "http://0.0.0.0:3001",
            "description": "Local test environment",
        },
    ],
    version="1.0.0",
)
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Explicitly enable parallelism

# Enable CORS for *
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("timing")
async def add_response_timing_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    end_time = time.time()
    response.headers["X-Response-Time"] = str((end_time - start_time) * 1000)
    return response


prefix = "/api/v1"
app.include_router(chatbot_router, prefix=prefix)


def handle_error_msg(request, error_msg, error_code=None):
    request_url = str(request.url)
    error_msg = f"client error in {request_url}: {error_msg}"
    logger.error(error_msg)
    result = error_msg.split(":")[-1].strip()
    return result


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    msg = exc.errors()[0]["msg"]
    error_msg = handle_error_msg(request, msg)
    return JSONResponse(
        status_code=400,
        content={
            "status_code": StatusCode.ERROR_INPUT_FORMAT,
            "msg": error_msg,
        },
    )


@app.exception_handler(asyncio.TimeoutError)
async def timeout_exception_handler(request: Request, exc: asyncio.TimeoutError):
    request_url = str(request.url)
    logger.error(f"TimeoutError: {exc} in request_url: {request_url}")
    return JSONResponse(status_code=200, content={
        "status_code": StatusCode.ERROR_TIMEOUT,
        "msg": f"TimeoutError during request url: {request_url}",
    })


def main(host="0.0.0.0", port=3001):
    # show if there is any python process running bounded to the port
    # ps -fA | grep python
    logger.info("Start api server")
    uvicorn.run("app.main:app", host=host, port=port)


if __name__ == "__main__":
    main()