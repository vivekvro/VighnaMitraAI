# Standard
import os
import dotenv

# Third-party
from psycopg import AsyncConnection
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
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
    retrieval_info_fetcher_node,
    retriever_node,
    retrieve_user_memory_node,
)
from src.chatbots.node_conditions import (
        retrieval_router,route_retrieval_type
    )
# --------------------------------------------------------------------------------------

dotenv.load_dotenv()

DB_POSTGRES_URL = os.getenv("DB_POSTGRES_URL")


async def base_chatbot():

    
    
    #  Build graph
    builder_graph = StateGraph(ChatBotState)


    
    #While only a few MCP tools are included, they demonstrate my capability to build MCP servers and implement custom tool logic
    _,tool_node = await initialize_mcp_tools()

    builder_graph.add_node("init_SystemMessage", init_SystemMessage)
    builder_graph.add_node("system_message_summarizer_node", system_message_summarizer_node)
    builder_graph.add_node("chat_node", chat_node)
    builder_graph.add_node("tool_node", tool_node)
    builder_graph.add_node("summarize_node", summarize_conversation)


    builder_graph.add_node("remember_node", remember_node)

    builder_graph.add_node("retrieval_info_fetcher_node",retrieval_info_fetcher_node)
    builder_graph.add_node("retriever_node", retriever_node)
    builder_graph.add_node("retrieve_user_memory_node", retrieve_user_memory_node)

    # Define edges
    builder_graph.add_edge(START, "init_SystemMessage")
    builder_graph.add_edge("init_SystemMessage", "system_message_summarizer_node")
    # Conditional edges from system_message_summarizer_node based on MemoryCondition
    builder_graph.add_conditional_edges("system_message_summarizer_node", retrieval_router, {
        "need_retrieval": "retrieval_info_fetcher_node",
        "chat_node": "summarize_node"
    })
    builder_graph.add_conditional_edges("retrieval_info_fetcher_node",route_retrieval_type,{
        "user_memories":"retrieve_user_memory_node",
        "uploaded_documents":"retriever_node"
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

    postgres_conn = await AsyncConnection.connect(
        conninfo=DB_POSTGRES_URL,
        autocommit=True
    )

    #  Store setup
    store = AsyncPostgresStore(conn=postgres_conn)
    checkpointer = AsyncPostgresSaver(conn=postgres_conn)

    await store.setup()
    await checkpointer.setup()

    #  Compile graph
    return builder_graph.compile(
        checkpointer=checkpointer,
        store=store
    )