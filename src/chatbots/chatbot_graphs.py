# Standard
import os
import dotenv

# Third-party

from psycopg import AsyncConnection
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.prebuilt import tools_condition,ToolNode

# Local
from src.state import ChatBotState
from src.rag.retrievers import postgres_embed
from src.chatbots.nodes import (
    
    init_SystemMessage,
    system_message_summarizer_node,
    chat_node,
    retrieval_evaluation_node,
    remember_node,
    mcp_tools,
    tools_trace_node,
    summarize_conversation,
    retrieval_info_fetcher_node,
    retriever_node,
    retrieve_user_memory_node,
)
from src.chatbots.node_conditions import (
        retrieval_router,retrieval_router_node
    )
# --------------------------------------------------------------------------------------

dotenv.load_dotenv()

DB_POSTGRES_URL = os.getenv("DB_POSTGRES_URL")


async def base_chatbot():

    
    
    #  Build graph
    builder_graph = StateGraph(ChatBotState)


    
    #While only a few MCP tools are included, they demonstrate my capability to build MCP servers and implement custom tool logic
    async with mcp_tools() as tools:
        tool_node = ToolNode(tools)

    builder_graph.add_node("init_SystemMessage", init_SystemMessage)
    builder_graph.add_node("system_message_summarizer_node", system_message_summarizer_node)
    builder_graph.add_node("chat_node", chat_node)
    builder_graph.add_node("tool_node", tool_node)
    builder_graph.add_node("tools_trace_node",tools_trace_node)
    builder_graph.add_node("summarize_node", summarize_conversation)


    builder_graph.add_node("remember_node", remember_node)
    builder_graph.add_node("retrieval_router_node",retrieval_router_node)
    builder_graph.add_node("retrieval_info_fetcher_node",retrieval_info_fetcher_node)
    builder_graph.add_node("retriever_node", retriever_node)
    builder_graph.add_node("retrieve_user_memory_node", retrieve_user_memory_node)
    builder_graph.add_node("retrieval_evaluation_node", retrieval_evaluation_node)

    # Define edges
    builder_graph.add_edge(START, "init_SystemMessage")
    builder_graph.add_edge("init_SystemMessage", "system_message_summarizer_node")
    # Conditional edges from system_message_summarizer_node based on MemoryCondition
    builder_graph.add_edge("system_message_summarizer_node","retrieval_router_node")
    builder_graph.add_conditional_edges("retrieval_router_node", retrieval_router, {
        "need_retrieval": "retrieval_info_fetcher_node",
        "chat_node": "summarize_node"
    })
    builder_graph.add_edge("retrieval_info_fetcher_node","retriever_node")
    builder_graph.add_edge("retrieval_info_fetcher_node","retrieve_user_memory_node")

    builder_graph.add_edge("summarize_node", "chat_node")
    # Conditional edges from chat_node based on tools_condition
    builder_graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tools_trace_node",
        "__end__": "remember_node"
    })
    builder_graph.add_edge("tools_trace_node", "tool_node")
    builder_graph.add_edge("tool_node", "chat_node")
    # feeding retrieved information back into the summarizer to ensure it has the most up-to-date context for generating summaries and guiding the conversation effectively.
    builder_graph.add_edge("retriever_node", "retrieval_evaluation_node")# feeding retrieved information back into the summarizer to ensure it has the most up-to-date context for generating summaries and guiding the conversation effectively.
    builder_graph.add_edge("retrieve_user_memory_node", "retrieval_evaluation_node")# feeding retrieved user memory back into the summarizer to ensure it has the most up-to-date context for generating summaries and guiding the conversation effectively.
    
    builder_graph.add_edge("retrieval_evaluation_node", "chat_node")


    builder_graph.add_edge("remember_node", END)

    postgres_conn_1 = await AsyncConnection.connect(
        conninfo=DB_POSTGRES_URL,
        autocommit=True
    )
    postgres_conn_2 = await AsyncConnection.connect(
        conninfo=DB_POSTGRES_URL,
        autocommit=True
    )

    #  Store setup
    store = AsyncPostgresStore(conn=postgres_conn_1,
    index={
        "embed": postgres_embed,
        "dims": 1024,
        "text_fields": ["data"]
    })
    checkpointer = AsyncPostgresSaver(conn=postgres_conn_2)

    await store.setup()
    await checkpointer.setup()

    #  Compile graph
    return builder_graph.compile(
        checkpointer=checkpointer,
        store=store
    )