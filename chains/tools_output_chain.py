from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm


_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an agent specialized in merging the responses obtained from tools into a single final answer to show the user.\n"
        "Take the responses obtained from the tools, merge them logically and coherently, and return a single clear and complete final answer to show the user.\n"
        "If the tool responses are incomplete or insufficient, still return the most complete answer possible based on the data obtained, without inventing anything.\n"
        "Never include the string 'tools_needed' in the final answer.\n"
        "RULES:\n"
        "- The final answer must be coherent and well structured.\n"
        "- The final answer must directly reference the original request.\n"
        "- The final answar must be in the same language of the request, TRANSLATE EVERYTHING IF IT IS NEEDED WITHOUT change the meaning\n"

    )),
    ("user", (
        "Here is the available information:\n\n"
        "Initial request: {original_text}\n\n"
        "Tool results:\n\n{tools_outputs}\n\n"
        "Merge them into a single final answer to show the user."
    ))
])

tools_output_chain = _prompt | llm | StrOutputParser()
