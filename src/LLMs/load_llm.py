from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
def llama_3_3_70b_versatile(temperature=0.7):
    load_dotenv()
    return ChatGroq(model="llama-3.3-70b-versatile" ,temperature=temperature)
def gpt_oss_120b(temperature=0.7):
    load_dotenv()
    return ChatGroq(model="openai/gpt-oss-120b",temperature=temperature)

def gpt_oss_20b(temperature=0.7):
    load_dotenv()
    return ChatGroq(model="openai/gpt-oss-20b",temperature=temperature)


def qwen3_32b(temperature=0.7):
    load_dotenv()
    return ChatGroq(model="qwen/qwen3-32b",temperature=temperature)

def gemma4_e4b(temperature=0.7):
    load_dotenv()
    return ChatOllama(model="gemma4:e4b",num_gpu=0,temperature=temperature)
def llama3_8b(temperature=0.7):
    load_dotenv()
    return ChatOllama(model="llama3:8b",num_gpu=0,temperature=temperature)
def qwen2_5_coder_7b(temperature=0.7):
    load_dotenv()
    return ChatOllama(model="qwen2-5-coder:7b",num_gpu=0,temperature=temperature)