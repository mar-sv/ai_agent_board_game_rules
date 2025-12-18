FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch


# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/app/src

# copy everything under src
COPY src/ ./src/

# optional: run your app
CMD ["python", "src/boardgame_agents/main.py"]
