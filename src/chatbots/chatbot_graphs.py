# Standard
import os
import dotenv

# Third-party
import psycopg
from aiosqlite import connect
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.postgres import PostgresStore
from langgraph.prebuilt import tools_condition

# Local
from src.state import ChatBotState
from src.chatbots.nodes import (
    init_SystemMessage,
    system_message_summarizer_node,
    chat_node,
    remember_node,
    initialize_mcp_tools,
    summarize_conversation,
    retriever_node,
    retrieve_user_memory_node,
)
from src.chatbots.node_conditions import (
        MemoryCondition
    )
# --------------------------------------------------------------------------------------

dotenv.load_dotenv()

DB_POSTGRES_URL = os.getenv("DB_POSTGRES_URL")


async def base_chatbot():

    #  Async Postgres connection with autocommit
    postgres_conn = psycopg.connect(
        conninfo=DB_POSTGRES_URL,
        autocommit=True
    )

    #  Store setup
    store = PostgresStore(conn=postgres_conn)
    store.setup()

    #  Build graph
    builder_graph = StateGraph(ChatBotState)

    _,tool_node = await initialize_mcp_tools()

    builder_graph.add_node("init_SystemMessage", init_SystemMessage)
    builder_graph.add_node("system_message_summarizer_node", system_message_summarizer_node)
    builder_graph.add_node("chat_node", chat_node)
    builder_graph.add_node("tool_node", tool_node)
    builder_graph.add_node("summarize_node", summarize_conversation)
    builder_graph.add_node("remember_node", remember_node)
    builder_graph.add_node("retriever_node", retriever_node)
    builder_graph.add_node("retrieve_user_memory_node", retrieve_user_memory_node)

    # Define edges
    builder_graph.add_edge(START, "init_SystemMessage")
    builder_graph.add_edge("init_SystemMessage", "system_message_summarizer_node")
    # Conditional edges from system_message_summarizer_node based on MemoryCondition
    builder_graph.add_conditional_edges("system_message_summarizer_node", MemoryCondition, {
        "retriever_node": "retriever_node",
        "retrieve_user_memory_node":"retrieve_user_memory_node",
        "chat_node": "summarize_node"
    })
    
    

    builder_graph.add_edge("summarize_node", "chat_node")
    # Conditional edges from chat_node based on tools_condition
    builder_graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": "remember_node"
    })
    # feeding retrieved information back into the summarizer to ensure it has the most up-to-date context for generating summaries and guiding the conversation effectively.
    builder_graph.add_edge("retriever_node", "system_message_summarizer_node")# feeding retrieved information back into the summarizer to ensure it has the most up-to-date context for generating summaries and guiding the conversation effectively.
    builder_graph.add_edge("retrieve_user_memory_node", "system_message_summarizer_node")# feeding retrieved user memory back into the summarizer to ensure it has the most up-to-date context for generating summaries and guiding the conversation effectively.
    builder_graph.add_edge("tool_node", "chat_node")

    builder_graph.add_edge("remember_node", END)


    #  Async SQLite checkpoint
    conn = await connect("data/vighnamitraai.db", check_same_thread=False)
    checkpointer = AsyncSqliteSaver(conn=conn)

    #  Compile graph
    return builder_graph.compile(
        checkpointer=checkpointer,
        store=store
    )