# YesTo: The AI that never says no.

## Package Used

- `dotenv` 
- `langchain_google_genai`
- `langchain_ollama`
- `langchain_core`
- `langgraph`
- `sqlalchemy`
- `pydantic`
- `typing_extensions`
- `psycopg`
- `psycopg2`

## Set-Up Instruction To Run 'YesTo' Locally:
Create python environment and install needed package:
- `python3 -m venv {environment_name}`
- `source environment_name/bin/activate`
- `pip3 install -r requirements.txt`

## Extra Commands
- Access to the postgres db inside the docker container with:
docker exec -it {container name} bash

- Run the following command inside container `apt-get update && apt-get install -y postgresql-16-pgvector`

- Access to db:
    docker exec -it {container name} psql -U {DB user name} -d {db name} <-- db name used to run db inside container

- Run the following query:
    CREATE EXTENSION IF NOT EXISTS vector;

- Run the following query ti check:
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

- Install python dependencies inside project:
pip3 install pgvector langchain-postgres sentence-transformers

- Run tests with coverage: 
pytest tests/ --cov=. --cov-report=term-missing

- Library needed:
    sudo apt-get install libportaudio2


## SPECIAL ISTRUCTION 
docker compose up --build -d psql-db my-ollama

docker compose exec my-ollama ollama pull llama3.1

docker compose run --rm yes-to

## RUN
- `python3 main.py`