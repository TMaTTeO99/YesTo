import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


# load environment variables
load_dotenv()

# init model
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm = ChatOllama(model="llama3.1", temperature=0)

def getDBAddress(): 

    db_user = os.getenv("DB_USER")
    db_passwd = os.getenv("DB_PASSWD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    
    return f"postgresql://{db_user}:{db_passwd}@{db_host}:{db_port}/{db_name}"