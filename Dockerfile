FROM python:3.9.18-bookworm

ENV PYTHONPATH=/ai-chatbot

# Working directory for application
WORKDIR /ai-chatbot

# Copy and install requirements
COPY ./requirements.txt /ai-chatbot/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /ai-chatbot/requirements.txt

# Copy application code
COPY ./app /ai-chatbot/app

# Expose the application port. 
EXPOSE 3001

# Default command to run the application    
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3001"]