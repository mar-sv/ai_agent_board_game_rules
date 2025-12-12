FROM python:3.11-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy everything under src
COPY src/ ./src/

# optional: run your app
CMD ["python", "src/boardgame_agents/main.py"]
