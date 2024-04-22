FROM python:3.9.18-bookworm

ENV PYTHONPATH=/app
# Working directory for application
WORKDIR /app

# Copy and install requirements
COPY ./requirements.txt /app/requirements.txt
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
# Copy application code
COPY . /app

# Expose the application port.
EXPOSE 3001

# Default command to run the application
CMD ["python3", "./hpc_react_agent_with_vector_storage_and_function_calling_.py"]