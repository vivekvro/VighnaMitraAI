# Standard
import os
import re
import time
import asyncio
import datetime as dt
from uuid import uuid4
from sqlite3 import connect


# Third-party
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Local
from src.encrypt import ComparePasswords
from src.user_auth import (
    create_accounts_info_table,
    insert_account_info,
    check_if_email_exists,
    check_if_user_exists,
    fetch_password_by_username,
)
from src.rag.retrievers import update_vectorstore
from src.chatbots.chatbot_graphs import base_chatbot
from src.rag.DocumentsLoader import load_tempfile_path, DocLoader

load_dotenv()




db_path = "data/vighnamitraai.db"

def load_chatbot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(base_chatbot())

chatbot = load_chatbot()







def create_timestamp(db_path):
    with connect(db_path) as con:
        cur = con.cursor()

        # ✅ check table exists first
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='checkpoints';
        """)
        if not cur.fetchone():
            return  # 🚀 skip safely

        # check columns
        cur.execute("PRAGMA table_info(checkpoints)")
        columns = [col[1] for col in cur.fetchall()]

        if "created_at" not in columns:
            cur.execute("""
                ALTER TABLE checkpoints 
                ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """)







create_timestamp(db_path)

def validate_username(username: str):
    pattern = r"^[a-zA-Z0-9._]{3,20}$"
    
    if not re.match(pattern, username):
        raise ValueError("Username must be 3-20 chars, only letters, numbers, '_' or '.' allowed")
    
    return True

def validate_email(email: str):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    if not re.match(pattern, email):
        raise ValueError("Invalid email format")
    
    return True

def validate_password(pw):
    if len(pw) < 8:
        return False
    if not re.search(r"[A-Z]", pw):
        return False
    if not re.search(r"[a-z]", pw):
        return False
    if not re.search(r"\d", pw):
        return False
    if not re.search(r"[!@#$%^&*]", pw):
        return False
    return True


def confirm_passwords(pwd,c_pwd):
    if pwd!=c_pwd:
        raise ValueError("Passwords do not match")
    return True


if "user" not in st.session_state:
    entrypoint = st.selectbox("SignUp/In",["New User","Existing User"])

    if entrypoint == "New User":
        st.subheader("Welcome")
        username = st.text_input("Username")
        dob = st.date_input(
            "date of birth",
            min_value=dt.date(1947,8,15),
            max_value=dt.date.today(),
            format="YYYY-MM-DD")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.button("**SignUp**"):
            try:
                validate_username(username=username)
                validate_email(email=email)
                if check_if_user_exists(username,db_path):
                    st.error("User already exists!")
                    st.stop()
                if check_if_email_exists(email,db_path):
                    st.error("Email already exists!")
                    st.stop()
                
                
                if not validate_password(pw=password):
                    st.error("Password must contain:\n- Minimum 8 characters\n- At least 1 uppercase letter (A-Z)\n- At least 1 lowercase letter (a-z)\n- At least 1 number (0-9)\n- At least 1 special character (!@#$%^&*)")
                    st.stop()
                confirm_passwords(password,confirm_password)
                insert_account_info(username=username,password=password,email=email,dob=dob,db_path=db_path)
                st.success(f"Account Created! hey {username.lower().strip()}.")
                st.session_state['user']={"username":username.lower().strip()}
                st.rerun()



            except ValueError as e:
                st.error(e)
                st.stop()
                
            
            

    elif entrypoint == "Existing User":
        st.subheader("Welcome Back :-)")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("SignIn"):
            
            try:
                if not username or not password:
                    st.warning("Please fill all fields")
                    st.stop()
                if not check_if_user_exists(username,db_path):
                        st.error("User does not exist!")
                        st.stop()
                stored_pwd = fetch_password_by_username(username=username,db_path=db_path)
                if not ComparePasswords(password,stored_pwd):
                    st.error("Invalid Password")
                    st.stop()
                
                st.session_state['user'] = {"username":username.lower().strip()}
                st.rerun()
            except ValueError as e:
                st.error(e)
                st.rerun()
                
            
    st.stop()






username = st.session_state['user']['username']

if "chat_id"  not in  st.session_state["user"]:
    st.session_state["user"]['chat_id']= f"{username}_{str(uuid4())}"


def is_chat_empty(thread_id:str):
    response = requests.get(url=f"http://localhost:8005/is_chat_empty?thread_id={thread_id}")
    return response.json()['response']['is_empty']



st.sidebar.title("Vighna Mitra Ai")
st.sidebar.markdown("---")
st.sidebar.header(f"Username: {username}")
st.sidebar.markdown("---")

if st.sidebar.button("New chat"):
    current_id = st.session_state["user"]["chat_id"]

    if is_chat_empty(current_id):
        st.sidebar.warning("Current chat is empty. Use it first.")
    else:
        st.session_state["user"]["chat_id"] = f"{username}_{str(uuid4())}"
        st.rerun()



response_threads = requests.get(url=f"http://localhost:8005/thread_ids?user_id={username}")
threads = response_threads.json()['response']['thread_ids']



st.sidebar.markdown("---\ncurrent chat:")
st.sidebar.button(st.session_state["user"]["chat_id"],width=200)
st.sidebar.markdown("---\n")
sidebar_sections = st.sidebar.selectbox("select: ",["chat history","connectors","attach documents"],width=200)
if sidebar_sections =="chat history":
    chat_history = threads
    st.sidebar.markdown("---\nchat history chat:")
    if chat_history:
        for chat in chat_history:
            if st.sidebar.button(f"{chat}"):
                st.session_state["user"]["chat_id"] = chat
                st.rerun()

elif sidebar_sections == "attach documents":
    doctype = st.sidebar.selectbox(
        "select document type", ["pdf", "txt", "url"], width=200
    )

   
    if doctype in ["pdf", "txt"]:
        temp_path = None
        uploaded_file = st.sidebar.file_uploader(
            "upload here", type=["pdf", "txt"]
        )

    else:
        uploaded_file = st.sidebar.text_input("Enter URL:")

    if st.sidebar.button("Upload"):
        
        with st.sidebar.spinner("Uploading..."):

                try:
                    
                    if doctype in ["pdf","txt"]:
                        path = load_tempfile_path(uploaded_file)
                        loader = DocLoader(doctype=doctype,path=path)
                        docs = loader.load()
                        if not update_vectorstore(docs=docs,user_id=st.session_state['user']['username']):
                            st.sidebar.error("Upload failed")
                            os.remove(path)
                        else :
                            st.sidebar.success("Uploaded Successfully")
                            os.remove(path)
                    elif doctype =="url":
                        path = uploaded_file
                        loader = DocLoader(doctype=doctype,path=path)
                        docs = loader.load()
                        if not update_vectorstore(docs=docs,user_id=st.session_state['user']['username']):
                            st.sidebar.error("Upload failed")
                        else :
                            st.sidebar.success("Uploaded Successfully")
                    
                except Exception as e:
                    st.sidebar.error(str(e))
                



elif sidebar_sections == "connectors":
    connectors_type = st.sidebar.selectbox("select MCP Server  type :",["online","local"])
    if connectors_type == "online":
        mcp_server_name= st.sidebar.text_input("server Name:")
        mcp_server_url = st.sidebar.text_input("server url:")
        st.session_state['user']["mcp"] = {
            "type":connectors_type,
            "server_info":{
                "name":mcp_server_name,
                "url":mcp_server_url
                }
                }
    if connectors_type =="local":
        if st.sidebar.button("Coming Soon."):
            with st.sidebar.spinner("......"):
                time.sleep(10)
                st.sidebar.success("Still you need to wait.")





st.sidebar.markdown("---")
if st.sidebar.button("logout"):
    if "user" in st.session_state:
        del st.session_state["user"]
    st.rerun()

config = {"configurable":{
            "user_id":username,
            "thread_id":st.session_state['user']['chat_id']
        }
    }
try:
    response_messages = requests.get(url=f"http://localhost:8005/chat/history?thread_id={st.session_state['user']['chat_id']}")
    if response_messages.status_code == 200:
        messages = response_messages.json()['response']['messages']
    else:
        messages=None
except:
    messages=None

if messages:
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.write(msg['content'])

user_input = st.chat_input("Ask Anything")



def fake_stream_response(text:str):
    for char in text:
        yield char
        time.sleep(0.002)



"""
# def get_chatbot_response(user_id:str,user_input:str,config: dict):
#     return asyncio.run(chatbot.ainvoke({
#         "messages":[HumanMessage(content=user_input)],
#         "system_messages": [],
#         "summary": {
#             "summary_content": "",
#             "summary_end_index": 0
#         },
#         "retrieval_details": None,
#         "user_details": {
#             "user_id": user_id,
#             "user_memory": None
#         },
#         "trace": []
#     },
#                 config=config
#             ))
"""

if user_input:
    with st.chat_message(name="user"):
        st.write(user_input)
    with st.spinner("thinking...."):
        response = requests.post(
            "http://localhost:8005/chat",
            json={
                "message":user_input,
                "user_id":username,
                "thread_id":st.session_state['user']['chat_id']
            }
        )
        result_state = response.json()['response']

    with st.chat_message(name="assistant"):
        if "trace" in result_state:
            with st.expander("⚙️ Execution Trace"):
                for step in result_state["trace"]:
                    st.write(step)
        st.write_stream(fake_stream_response(result_state['messages'][-1].content))

