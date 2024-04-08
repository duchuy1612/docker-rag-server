FROM python:3.9.18-bookworm
# Set environment variables for CMake
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
ENV FORCE_CMAKE=1

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY . .
EXPOSE 3001

CMD ["python3", "-m" , "./hpc_react_agent_with_vector_storage_and_function_calling_.py"]