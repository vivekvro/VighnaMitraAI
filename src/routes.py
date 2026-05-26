from fastapi import FastAPI,HTTPException,UploadFile
from src.rag.DocumentsLoader import DocLoader
from pydantic import BaseModel,Field
from typing import Annotated,Literal
from src.rag.retrievers import update_vectorstore
from src.chatbots.chatbot_graphs import base_chatbot
from langchain_core.messages import HumanMessage
from sqlite3 import connect
import os
from contextlib import asynccontextmanager
DB_PATH = "data/vighnamitraai.db"

from dotenv import load_dotenv
load_dotenv()

chatbot = None







@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot
    chatbot = await base_chatbot()
    print("Chatbot loaded!.")
    yield
    print("APP shutdown")









#-------------------------------------------------------------------------------

class FileDetails(BaseModel):
    path:Annotated[str,Field(description="Path to the document. Can be a URL or a local file path (e.g., from tempfile).")]
    doctype:Literal['pdf','txt','url']
    user_id:str

app =  FastAPI(lifespan=lifespan)



@app.post("/upload_document")
def get_upload_docs(file:FileDetails):
    try:
        loader = DocLoader(doctype=file.doctype,path=file.path)
        docs = loader.load()
        if not docs:
            raise HTTPException(status_code=500,detail="NO document is loaded")
        if update_vectorstore(docs=docs,user_id=file.user_id):
            return {"response":"Uploaded Successfully"}
        else:
            return {"response":"something went wrong."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class UserDetails(BaseModel):
    message:str =Field(description="User's Query/Message")
    user_id: str = Field(description="User's user_id")
    thread_id:str = Field(description="Thread_id")

@app.post("/chat")
async def post_chat_response(req:UserDetails):
    global chatbot
    try:
        response = await chatbot.ainvoke({
            "messages":[HumanMessage(content=req.message)],
            "system_messages": [],
            "summary": {
                "summary_content": "",
                "summary_end_index": 0
            },
            "retrieval_details": None,
            "user_details": {
                "user_id": req.user_id,
                "user_memory": None
            },
            "trace": []
        },config={"configurable":{
            "thread_id":req.thread_id,
            "user_id":req.user_id}})
        return {
            "response":response["messages"][-1].content
            }
    except Exception as e:
        return {"error":str(e)}

@app.get("/chat/history")
async def get_history_messages(thread_id:str):
    try:
        state = await chatbot.aget_state(config={"configurable":{"thread_id":thread_id}})
        messages =  state.values.get("messages",[])
        returnable_messages = []
        for msg in messages:
            role = "user" if msg.type=="human" else "assistant"
            content = msg.content
            returnable_messages.append(
                {
                    "role":role,
                    "content":content
                }
            )
        return {"response":
                {
                    "thread_id":thread_id,
                    "messages":returnable_messages
                    }
        }
    except Exception as e:
        return HTTPException

@app.get("/thread_ids")
def get_thread_ids(user_id: str):
    if not os.path.exists(DB_PATH):
        return []
    with connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='checkpoints';
            """)
        if not cur.fetchone():
            return []
        cur.execute("""
                    SELECT DISTINCT thread_id FROM checkpoints
                    WHERE thread_id LIKE ?
                    ORDER BY created_at DESC
                    """,(f"{user_id}%",))
        rows = cur.fetchall()
    thread_ids = [r[0] for r in rows]
    return {
        "response":{
            "user_id":user_id,
            "thread_ids":thread_ids
            }
        }



