from fastapi import FastAPI,HTTPException,UploadFile,File,Form
from src.rag.DocumentsLoader import DocLoader
from pydantic import BaseModel,Field
from src.rag.retrievers import update_vectorstore
from src.chatbots.chatbot_graphs import base_chatbot
from langchain_core.messages import HumanMessage

from psycopg import AsyncConnection

import tempfile,os
from contextlib import asynccontextmanager
DB_PATH = os.getenv("DB_POSTGRES_URL")

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


app =  FastAPI(lifespan=lifespan)


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
            "response":{
                "message":response["messages"][-1].content,
                "trace":response['trace']
                }
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
async def get_all_threads(user_id: str):
    async with await AsyncConnection.connect(DB_PATH) as con:
        async with con.cursor() as cur:
            await cur.execute("""
                SELECT DISTINCT thread_id
                FROM checkpoints
                WHERE thread_id LIKE %s
                ORDER BY thread_id
            """, (f"{user_id}%",))

            rows = await cur.fetchall()

    return {
        "response":{
            "thread_ids": [thread_id for (thread_id,) in rows]
            }
        }




@app.get("/is_chat_empty")
async def is_chat_empty(thread_id: str):
    async with await AsyncConnection.connect(DB_PATH) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT 1
                FROM checkpoints
                WHERE thread_id = %s
                LIMIT 1
                """,
                (thread_id,)
            )

            row = await cur.fetchone()

    return {
        "response":{
            "is_empty": row is None
            }
        }



#---------------------------------------------
MIME_TO_EXTENSION = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/csv": "csv",
    "image/png": "png",
    "image/jpeg": "jpg",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

async def load_tempfile_path(upload_file:UploadFile):
    with tempfile.NamedTemporaryFile(delete=False,suffix=f"_{upload_file.filename}") as tmpfile:
        tmpfile.write(await upload_file.read())
        return tmpfile.name

@app.post("/upload")
async def upload(file:UploadFile=File(...),thread_id:str=Form(...),user_id:str=Form(...)):
    try:
        file_type= MIME_TO_EXTENSION[file.type]
        temp_fila_path = await load_tempfile_path(file)
        loader = DocLoader(doctype=file_type,path=temp_fila_path)
        docs = loader.load()

        if not docs:
            raise HTTPException(status_code=500,detail="NO document is loaded")
        if update_vectorstore(docs=docs,user_id=file.user_id):
            return {"response":"Uploaded Successfully"}
        else:
            return {"response":"something went wrong."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


