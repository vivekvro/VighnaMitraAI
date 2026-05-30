# Standard
from typing import Literal,Optional,List,Set
# Third-party
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
# Local
from src.state import ChatBotState
from src.LLMs.load_llm import llama3_4b


llm = llama3_4b()

#-------Memory_fetcher_condition------------
class FetchUserMemoryDetails(BaseModel):
    search_query:str = Field(
        description="""
Optimized query for memory retrieval.

Focus on user-related context such as:
preferences, habits, goals, projects, skills,
or conversational continuity.

Keep the query concise and retrieval-focused.
"""
    )
    filter_by_type:Literal[
        "personal", "habit","interests","goals","skills","dislikes", "preferences","learning_style",
        "projects","tools","constraints","knowledge_level","career","education","behavior",
        "decisions","context","health"
        ] = Field(
        description="""
Single memory category to filter retrieval.

Use the most relevant category for this query.

Examples:
- "preferences"
- "projects"
- "habit"
"""
    )
    num_docs:int = Field(
        default=10,
        ge=4,
        le=25,
        description="""
Number of memories to retrieve.

Memories are small atomic facts,
so larger retrieval sizes are acceptable.

Guidelines:
- 4-8: focused retrieval
- 9-15: normal conversational continuity
- 16-25: broad contextual personalization
"""
    )

class FetchUploadedDocsDetails(BaseModel):
    search_query:str = Field(
        description="""
Optimized semantic search query for uploaded documents.

Preserve important entities, concepts, and intent
while removing unnecessary conversational wording.
"""
    )
    retrieval_mode :Literal["similarity","mmr"]= Field(
        default="similarity",
        description="""
Retrieval strategy.

- "similarity":
  Best for highly relevant chunks.

- "mmr":
  Best for diverse retrieval with less redundancy.
""")
    filter_by_source:Optional[str] = Field(
        default=None,
        description="""
Optional source/document filter.

Restrict retrieval to a specific uploaded file,
document, URL, or knowledge source.

Examples:
- "ml_notes.pdf"
- "www.example.com"
- "notesofxyz.txt"
- "semester_notes"
"""
    )
    num_docs:int = Field(
            default=7,
            ge=4,
            le=15,
            description="""
    Number of document chunks to retrieve.
    
    Guidelines:
    - 4-6: precise factual retrieval
    - 7-10: explanatory or moderate complexity
    - 11-15: broad or multi-step reasoning
    """)

class MemoryCondition_decisions(BaseModel):
    user_query:Optional[str] = Field(description="""
The EXACT original latest user message.

This field MUST preserve the user's raw query exactly as written,
including:
- wording
- tone
- grammar mistakes
- spelling mistakes
- punctuation
- formatting
- conversational phrasing

DO NOT:
- rewrite
- summarize
- optimize
- clean
- expand
- interpret
- simplify
- convert into a retrieval query

This field is used for:
- conversational continuity
- debugging
- observability
- agent routing
- traceability
- preserving original user intent

Examples:

User says:
"waht backend framework i mostly use ?"

Return:
"waht backend framework i mostly use ?"

NOT:
"What backend framework do I usually use?"

User says:
"search my pdf for bert fine tuning"

Return:
"search my pdf for bert fine tuning"

NOT:
"Find BERT fine-tuning information in uploaded documents."
""")

    user_memories_retrieval_details: Optional[
        List[FetchUserMemoryDetails]
    ] = Field(
        default=None,
        description="""User-memory retrieval execution plans.

Use this field ONLY when personalized retrieval
from long-term user memory is required.

Examples:
- preferences
- habits
- goals
- projects
- learning style
- conversational continuity

Rules:
- Each retrieval object should focus on ONE category
- Prefer focused retrieval plans over broad queries
- Keep queries short and intent-focused
- Multiple retrieval plans are allowed

Return null when memory retrieval is unnecessary.
"""
    )
    uploaded_documents_retrieval_details: Optional[
        List[FetchUploadedDocsDetails]
        ] = Field(
        default=None,
        description="""Document retrieval execution plans.

Use this field ONLY when retrieval from uploaded
documents, PDFs, notes, URLs, or vector databases
is required.

Rules:
- Each object represents one retrieval strategy
- Multiple retrieval plans are allowed
- Keep search queries concise and semantic
- Use filter_by_source only when useful
- Prefer similarity for precision
- Prefer mmr for broader context diversity

Return null when document retrieval is unnecessary.
"""
    )

def MemoryCondition(state:ChatBotState):
    messages = state['messages']
    system_message = state['system_messages']
    summary = state['summary']['summary_content']
    parser = PydanticOutputParser(pydantic_object=MemoryCondition_decisions)
    if len(messages)>1 and not summary:
        conversation_msgs = messages[:-1]
        conversation = "\n".join(
    [
        f"{'human' if isinstance(msg, HumanMessage) else 'ai'}: {msg.content}"
        for msg in conversation_msgs
        if isinstance(msg, (HumanMessage, AIMessage))
    ]
)
    elif len(messages)>1 and summary:
        last_idx = state['summary']['summary_end_index']
        conversation_msgs = messages[last_idx:-1]
        conversation = "\n".join(
    [
        f"{'human' if isinstance(msg, HumanMessage) else 'ai'}: {msg.content}"
        for msg in conversation_msgs
        if isinstance(msg, (HumanMessage, AIMessage))
    ]
)
        conversation = "(summary of previous conversation)"+"\n"+summary+"(later conversation)\n"+conversation
    elif len(messages)<=1:
        conversation = "No Conversation History yet"
    prompt = PromptTemplate(
    template="""
You are a retrieval planning system.

Your job:
Analyze the conversation and determine whether external retrieval is required.

You must decide:
- whether retrieval is needed
- which retrieval source to use
- how retrieval should be performed


When Retrieval IS Required
----------------
Set requires_retrieval = True when the answer depends on:
- uploaded PDFs or files
- URLs or knowledge bases
- vector database content
- long-term user memories
- past preferences, habits, goals, or projects
- information not available in the current conversation
- factual document-grounded answers
- personalized continuity from stored memories

Examples:
- "What was written in my uploaded PDF?"
- "Summarize my notes"
- "What backend framework do I usually use?"
- "What are my learning preferences?"
- "Search my uploaded documents for transformers"

When Retrieval is NOT Required
----------------
Set requires_retrieval = False when the query can be answered using:
- general knowledge
- reasoning or logic
- coding knowledge
- current conversation context
- simple calculations
- tool usage
- basic explanations
- latest public knowledge not dependent on uploaded docs
- casual conversation

Examples:
- "What is Python?"
- "Explain transformers"
- "2 + 2"
- "Write a FastAPI example"
- "Hello"

Retrieval Types
----------------

1. uploaded_documents
Use when information should come from:
- PDFs
- uploaded files
- notes
- URLs
- vector databases
- knowledge bases

Rules:
- retrieval_details must contain ONLY FetchUploadedDocsDetails objects
- Use concise semantic search queries
- Use filter_by_source only if clearly useful
- Prefer similarity for precise retrieval
- Prefer mmr for broader or diverse retrieval

2. user_memories
Use when personalization or past user context is needed.

Examples:
- preferences
- habits
- projects
- goals
- skills
- learning style
- conversational continuity

Rules:
- retrieval_details must contain ONLY FetchUserMemoryDetails objects
- Each retrieval object should focus on ONE memory category
- Prefer multiple focused retrieval plans over broad retrieval
- Keep memory queries short and intent-focused

General Rules
----------------
- Return ONLY valid structured output
- Do NOT explain reasoning
- Do NOT generate extra text
- Keep queries concise and retrieval-optimized
- Use smaller num_docs for precise retrieval
- Use larger num_docs for broader reasoning or personalization
- Avoid unnecessary retrieval
- If requires_retrieval = False:
    - retrieval_details must be null

Additionally:
- user_query MUST contain the exact raw latest user message
- Never optimize or rewrite user_query
- Retrieval search queries SHOULD be optimized separately
- Preserve the original user intent exactly
- user_query and retrieval search queries serve different purposes

{format_instructions}



system_message:
{system_message}
if information is available in system message then do not go for any retrieval and just answer the query based on system message information and conversation context without any retrieval.

Conversation:
{conversation}

Latest User Query:
{query}
""",
    input_variables=[
        "conversation",
        "system_message",
        "query"
    ],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)
    chain = prompt | llm | parser
    response = chain.invoke({
            "conversation":conversation,'system_message':system_message,
            "query":messages[-1].content})
    
    if response.requires_retrieval:
        if not response.user_query:
            state['retrieval_details']['user_msg'] = state['messages'][-1].content
        else:
            state['retrieval_details']['user_msg'] = response.user_query
        routes = []
        retrieval_type =response.retrieval_type or []
        if retrieval_type:
            if "user_memories" in retrieval_type:
                if response.user_memories_retrieval_details:
                    state['retrieval_details']['user_memories'] = response.user_memories_retrieval_details
                    routes.append("retrieve_user_memory_node")

            if "uploaded_documents" in retrieval_type:
                if response.uploaded_documents_retrieval_details:
                    state['retrieval_details']['rag_details'] = response.uploaded_documents_retrieval_details
                    routes.append("retriever_node")
            else:
                return "chat_node"
            if len(routes)==1:
                return routes[0]
            return routes

    else:
        return "chat_node"








class Memory_Router_Condition(BaseModel):
    requires_retrieval: bool = Field(
        description="""
Return True if answering the user's query requires retrieval from:
- uploaded documents
- URLs
- vector databases
- long-term user memories

Return False if the query can be answered using:
- general knowledge
- reasoning or logic
- conversation context
- tool usage
- latest web knowledge (not from uploaded docs)
- if information is already is available 
"""
    )

    retrieval_type:Optional[Set[Literal["uploaded_documents","user_memories"]]] = Field(
        description="""
Select the retrieval source type.

Options:
- "uploaded_documents":
  Use retrieval over uploaded files, PDFs, notes,
  URLs, or vector-store document chunks.

  In this case, retrieval_details MUST contain:
  List[FetchUploadedDocsDetails]

- "user_memories":
  Use retrieval over long-term user memories
  stored in BaseStore/PostgresStore.

  In this case, retrieval_details MUST contain:
  List[FetchUserMemoryDetails]
"""
    )

def memory_router(state: ChatBotState):
    parser=PydanticOutputParser(pydantic_object=Memory_Router_Condition)
    prompt = PromptTemplate(
        template="""You are a retrieval router.

Determine whether answering the user's query requires retrieval.

user's query:{user_query}

Available sources:

* `uploaded_documents`: user-uploaded files, PDFs, notes, URLs, and vector-store document chunks.
* `user_memories`: long-term stored user memories.

Set `requires_retrieval = True` only if the required information is not available from:

* the current conversation,
* the user's message,
* general knowledge,
* reasoning,
* tool outputs already available.

Use:

* `uploaded_documents` when the user refers to uploaded files, documents, notes, PDFs, URLs, or ingested knowledge.
* `user_memories` when the user asks about stored preferences, past decisions, goals, projects, or remembered information.
* both when both sources are needed.

Set `requires_retrieval = False` if the query can be answered directly without retrieval.

Prefer `False` when uncertain.

{format_instructions}
""",
        input_variables=['user_query'],
        partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
    )
    chain = prompt | llm | parser
    result = chain.invoke({"user_query":state['messages'][-1]})

    if not result.requires_retrieval:
        return_path = []
        retrieval_type = result.retrieval_type
        if "user_memories" in retrieval_type:
            return_path.append("user_memories")
        if "uploaded_documents" in retrieval_type:
            return_path.append("uploaded_documents")
        if len(return_path)==1:
            return return_path[0]
        return return_path
    else:
        return "chat_node"
