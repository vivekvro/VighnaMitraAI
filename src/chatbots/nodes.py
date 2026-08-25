# Standard Library
import os, asyncio, datetime, dotenv
from uuid import uuid4
from psycopg import AsyncConnection
from contextlib import asynccontextmanager
from typing import List,Optional,Literal
# Third-party Libraries
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage,AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.vectorstores import VectorStore
from langchain_core.runnables import RunnableConfig
# Local Project Imports
from src.LLMs.load_llm import llama_3_3_70b_versatile, gpt_oss_120b
from src.state import ChatBotState
from src.rag.retrievers import load_vectorstore,embedding,postgres_embed
from src.configs.config_methods import load_config
# =======================

dotenv.load_dotenv()
DB_POSTGRESSTORE_PATH = os.getenv("DB_POSTGRES_URL")
#----------------LLMs Setups -------------------------
llm_summarizer = llama_3_3_70b_versatile()# you can choose any summarization-capable model here, ideally a smaller one for efficiency, since summarization doesn't require the full power of a 70b model. Adjust based on your specific needs and token limits.
llm = gpt_oss_120b()# i am using this for token size efficiency, but you can choose any capable model here, ideally the same one used for the main conversation to maintain consistency in response style and capabilities. Adjust based on your specific requirements and token limits.

# tool setup

@asynccontextmanager
async def mcp_tools():
    mcp_config = await load_config()
    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools()
    yield tools





#------------------- trace  ---------------------


def update_trace(state,node_name:str |list[str]):
    if isinstance(node_name, list):
        return state['trace'] + node_name
    return state['trace'] + [node_name]


# current date & time

def get_current_date():
    return str(datetime.datetime.today()).split(" ")
#------------------- init System Message memory ------------------------
SYSTEM_PROMPT_TEMPLATE = """
    Basic info:
    - datetime: {datetime}
    - user_id: {user_id}
You are VighnaMitra, a knowledgeable and helpful AI assistant.

Your communication style:
- Answer concisely — get to the point, avoid unnecessary elaboration
- Be direct and clear in your language
- When information feels tangentially related but not directly applicable,
  treat it as unavailable rather than forcing a connection
  Simply state: "I don't have information on that" or similar

Tools and resources:
- Use tools (search, retrieval, calculation, etc.) when they provide
  genuine value to answering the query
- Avoid using tools for queries that can be answered from your knowledge
- Always prefer the most reliable and current source available

User context and memories:
- You have access to curated memories of the user's preferences,
  habits, goals, and past decisions
- Integrate this context naturally into your responses without
  explicitly mentioning it (don't say "I remember you like X",
  just acknowledge it through your answer choices and framing)
- Use these memories to personalize your approach, tone, and examples
- Only reference user history when it's genuinely relevant to answering
  the current question

Tone:
- Respectful and helpful
- Patient with follow-up questions
- Honest about limitations and uncertainty
- Avoid over-explaining or being patronising

Your role:
Answer the user's question thoroughly but efficiently, drawing on your
knowledge, available tools, and their personal context where appropriate.


    User memory:
    {user_details_content}
"""
async def get_BasicMemories(namespace: tuple,filter_by_type:str,search_query:str,num_docs:int,store: BaseStore):# This function retrieves basic memories from the vector store based on the provided namespace, filter type, search query, and number of documents to fetch. It constructs a search query using the specified parameters and retrieves relevant memories that match the filter type. The retrieved memories are then formatted into a string that can be included in the system message for initializing the conversation context.
    items = await store.asearch(
        namespace,
        query=search_query,
        limit=num_docs,
        filter={
            "type": filter_by_type
            })
    fetch_data = "\n".join([f"- {mem.value['data']} date: {mem.value['date']}" for mem in items  ])if items else "(No Memory exist)"
    return f"({filter_by_type})\n" +fetch_data

async def init_SystemMessage(state: ChatBotState, store: BaseStore):
    # Initialize the system message with basic user information,
    # relevant memories, and core behavioral instructions for the LLM
    # to guide the conversation from the very beginning.
    user_id = state["user_details"]["user_id"]

    if state['system_messages']:
        return state
    namespace = ("user",user_id,"details")
    memory_queries = {
        "personal": [
        "What is the user's name?",
        "What basic personal details are known about the user?"
    ],
    "habit": [
        "What are the user's daily habits or routines?",
        "What recurring behaviors does the user follow?",
        "What productivity or study habits does the user have?"
    ],
    "interests": [
        "What topics is the user interested in?",
        "What technologies or fields does the user enjoy learning?"
    ],
    "goals": [
        "What are the user's current goals?",
        "What career or learning goals does the user have?",
        "What is the user trying to achieve?"
    ],
    "skills": [
        "What skills does the user already have?",
        "What technical skills is the user learning?",
        "What tools or technologies is the user skilled in?"
    ],
    "dislikes": [
        "What does the user dislike?",
        "What types of responses or behaviors does the user prefer to avoid?",
        "What recommendations should not be repeated?"
    ],
    "preferences": [
        "What communication preferences does the user have?",
        "What response style does the user prefer?",
        "What formatting or explanation preferences does the user have?"
    ],
    "learning_style": [
        "How does the user prefer to learn?",
        "Does the user prefer hints, examples, or direct answers?",
        "What teaching style works best for the user?"
    ],
}
    all_memories = []

    for key, queries in memory_queries.items():

        search_query = " ".join(queries)

        result = await get_BasicMemories(
            namespace=namespace,
            filter_by_type=key,
            search_query=search_query,
            num_docs=5,
            store=store
        )

        all_memories.append(result)

    total_memories = "\n\n".join(all_memories)


    prompt = f"""
You are a user memory synthesiser.

You will receive a collection of retrieved user memories,
each with a timestamp or date context.

Your job is to compress them into a single coherent summary
that captures:
- The user's core preferences, habits, and goals
- Evolving patterns across time (e.g. "shifted from X to Y in March")
- Recent vs. established patterns (what's new vs. what's stable)
- Decisions or milestones that shaped the user's direction

Rules:
- Maximum 650 characters — hard limit
- Every detail must be actionable or informative for an AI assistant
- Drop obvious, redundant, or low-signal memories
- Preserve dates/timelines only when they matter (e.g. "recently started",
  "has been doing for 2 years") — don't list every timestamp
- Write in third person, present tense ("User prefers...", "User is building...")
- Highlight contradictions or shifts (e.g. "switched from tool A to tool B")
- Flag any unresolved goals or work-in-progress items

Output format:
Write ONLY the summary — no labels, no preamble, no markdown.
Make it dense and direct so it can be injected into an assistant's context.

Memories (with dates):
{total_memories[:6000]}
"""
    response = await llm_summarizer.ainvoke(prompt)

    system_message = SYSTEM_PROMPT_TEMPLATE.format(
        datetime=get_current_date()[0],
        user_id=user_id,
        user_details_content=response.content
        )

    return {
        'system_messages': [SystemMessage(
            content=system_message
        )]
    }


#------------------------ Chat node -----------------------------
# This node is responsible for generating the chatbot's response based on the conversation history, system messages, and any relevant context. It constructs the input messages for the LLM by combining the system messages, a summary of the conversation if available, and the recent messages since the last summary. It then invokes the LLM (with tools if necessary) to generate a response, which is returned as part of the updated state along with a trace of the nodes executed.
async def chat_node(state: ChatBotState):
    trace = update_trace(state,"Chat Node")
    last_summarized_index = state['summary']['summary_end_index']
    last_messages = state['messages'][last_summarized_index:]
    system_messages = state['system_messages']
    user_memories = state["user_details"]["user_memory"]
    messages = []

    # system
    messages.extend(system_messages)
    if state['summary']['summary_content']:
        messages.append(SystemMessage(
            content=f"last Conversation Summary:\n{state['summary']['summary_content']}"
        ))
        messages.extend(last_messages)
    else:
        messages.extend(last_messages)

    if state["retriever_context_message"]:
        context_message = state["retriever_context_message"]
        retriever_prompt = SystemMessage(content="""You are a retrieval consolidation system.

You are given:
1. The user's original query
2. Multiple RAG retrieval results generated from uploaded documents

Your task:
- Analyze all retrieval results together
- Identify useful, relevant, and non-contradictory information
- Combine related information into a single coherent response
- Ignore duplicate, noisy, weakly related, or irrelevant retrieval outputs
- Prefer information that directly answers the user's query
- Keep the final response concise but complete

Important Rules:
- Use ONLY information present in the RAG results
- Do NOT invent or assume missing information
- Do NOT mention retrieval systems, chunks, embeddings, or vector stores
- Do NOT mention which query retrieved which information
- If multiple retrievals contain overlapping information, merge them naturally
- If ALL retrieval results indicate missing/unrelated information, return exactly:

"No information related to your query is available in the uploaded documents."
""")
        response = await llm.ainvoke(messages + [retriever_prompt + context_message])
        return {
        "messages": [response],
        "retrieval_type":[],
        "retrieval_details":{
            "user_msg":"",
            "rag_details":[],
            "user_memories":[]
        },
        "retriever_context_message": None,
        "trace": trace
    }

    

    async with mcp_tools() as tools:
        llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke(messages)

    return {
        "messages": [response],
        "trace": trace
    }

def tools_trace_node(state: ChatBotState):
    tool_calls =state["messages"][-1].tool_calls
    if tool_calls:
        tool_names = [call['name'] for call in tool_calls]
        trace = update_trace(state,tool_names)
        return {"trace": trace}
    return {
        "trace":state['trace']
    }
#------------------------ summary Nodes ---------------------------
async def system_message_summarizer_node(state: ChatBotState):
    system_messages = state['system_messages']
    if system_messages:
        if sum(len(msg.content) for msg in system_messages) > 1800:
            trace = update_trace(state,"System Message Summarizer Node")
            system_content = "\n".join([msg.content for msg in system_messages])
            prompt = f"""
Summarize the following system messages into a concise format that retains all important instructions, user details, and context.
The summary should be clear and comprehensive while being as brief as possible.
Focus on preserving critical information that guides the chatbot's behavior and responses.
and ensure the final summary stays under 700 words to allow room for future context and conversation history.
remove any redundant, repetitive, or non-essential information while keeping the core instructions intact.
just return the summary without any explanations or formatting.
System Messages:
{system_content}

"""
            response = await llm_summarizer.ainvoke(prompt)
            return {"system_messages":[SystemMessage(content=response.content)],"trace": trace}
        else:
            return state

async def summarize_conversation(state: ChatBotState):
    last_summarized_index = state['summary']['summary_end_index']
    messages = state["messages"][last_summarized_index:]
    if len(messages) > 20  or  count_tokens_approximately(messages) > 1800:
        trace =  update_trace(state,"History Conversation Summarizer Node")

        if len(messages) > 20:
            chunk = messages[:20]
            new_summarized_index= last_summarized_index+20
        else:
            chunk = messages
            new_summarized_index = last_summarized_index + len(chunk)



        existing_summary = state['summary']['summary_content']

        if existing_summary:# If there's already an existing summary, we want to update it with the new conversation chunk. The prompt will instruct the model to retain important information and keep the summary concise, ensuring it stays within a reasonable token limit for future context.
            prompt = (
                f"Existing summary:\n{existing_summary}\n\n"
                "Update this summary using the new conversation above. "
                "Keep it concise, and retain only important information relevant for future conversation context. "
                "Ensure the final summary stays under 600-700 words "
                "Avoid repetition and unnecessary details."
            )
        else:
            prompt = (
                "Summarize the conversation above concisely. "
                "Include only important information relevant for future conversation context. "
                "Ensure the summary stays under 600-700 words. "
                "Avoid repetition and unnecessary details."
            )

        # use full conversation for summarization
        messages_for_summary =[SystemMessage(content=prompt)] + chunk

        response = await llm_summarizer.ainvoke(messages_for_summary)

        return {
            "summary":{
                "summary_content":response.content,
                "summary_end_index":new_summarized_index},
            "trace": trace
        }
    else:
        return state

#------------------Remember-node-----------------------------
class NewMemoryDetails(BaseModel):# This Pydantic model defines the structure for new memory details that are extracted from the conversation. Each memory consists of a concise, atomic fact (memory) and a corresponding category (memory_type) that classifies the type of information. The memory_type is restricted to specific categories such as personal, habit, interests, goals, skills, dislikes, preferences, learning_style, projects, tools, constraints, knowledge_level, career, education, behavior, decisions, context, and health. This structured format ensures that the extracted memories are organized and can be easily stored and retrieved for future use in the conversation.
    memory:str = Field(default_factory=str,description="Only new long-term memory,No explanation.")
    memory_type: Literal[
        "personal", "habit","interests","goals","skills","dislikes", "preferences","learning_style",
        "projects","tools","constraints","knowledge_level","career","education","behavior",
        "decisions","context","health",
] = Field(
    description="""
Categorize the type of long-term memory extracted from the user.

Use:
- "personal": Identity details (name, background, location, etc.)
- "habit": Repeated behaviors or routines
- "interests": Topics, domains, or activities the user likes
- "goals": Short-term or long-term objectives
- "skills": Abilities, expertise, or things the user knows
- "dislikes": Things the user avoids or does not like
- "preferences": Communication or response preferences (tone, length, format)
- "learning_style": How the user prefers to learn (examples, hints, step-by-step, etc.)
- "projects": Ongoing or recurring work the user is involved in
- "tools": Technologies, frameworks, or tools the user uses
- "constraints": Limitations such as time, device, resources, or restrictions
- "knowledge_level": User’s proficiency level in a specific domain
- "career": Job aspirations, professional direction, or industry focus
- "education": Academic background, degree, or subjects studied
- "behavior": Behavioral patterns (e.g., consistency, procrastination)
- "decisions": Important choices made by the user that affect future context
- "context": Temporary but reusable situations (e.g., exam prep, current focus area)
- "health": Health-related info ONLY if explicitly shared and safe to store

Rules:
- Choose the single best matching category.
- Do not create new categories.
- Prefer more specific types over generic ones.
- Avoid storing sensitive data unless explicitly allowed (especially for "health").
""")

class MemoryDecision(BaseModel):    # This Pydantic model defines the structure for making decisions about which memories to remember from the conversation. It includes a field to indicate whether new information should be remembered and a list to store the newly extracted memories.
    need_to_remember :bool= Field(description="""
Return True only if the conversation includes persistent, reusable information(e.g., preferences, identity, goals, constraints, or important context).
Return False if the content is generic, one-time, or not useful for future conversations.
- If False, new_memories MUST be an empty list.
- If True, new_memories MUST contain at least one valid memory.
""")
    new_memories: Optional[List[NewMemoryDetails]] = Field(
        default_factory=list,
        description="""List of newly extracted long-term memories from the current conversation.

Guidelines:
- Include ONLY new, relevant, and reusable information.
- Each item must be concise, atomic (one fact per entry), and self-contained.
- Do NOT include explanations, reasoning, or extra text—only the memory itself.
- Avoid duplicating existing memories.
- Skip trivial, one-time, or non-useful information.
- Ensure the memory is meaningful for improving future interactions (e.g., preferences, goals, habits, projects, constraints).

Formatting:
- Write memories in clear, normalized form (e.g., "User prefers short responses").
- Do not include timestamps, metadata, or conversational phrases.

If no valid memory is found, return an empty list.
""")

async def remember_node(state: ChatBotState, store: BaseStore):# This node is responsible for determining whether there is new, reusable information in the recent conversation that should be remembered for future interactions. It retrieves the existing memories for the user, analyzes the recent messages using a structured prompt to decide if new memories should be extracted, and if so, it stores the new memories in the vector store. The decision and extraction process is guided by specific rules to ensure that only relevant and useful information is remembered, while avoiding trivial or one-time details.

    user_id = state['user_details']["user_id"]
    namespace = ("user", user_id, "details")

    # 🔹 1. Fetch existing memory safely
    items = await store.asearch(namespace,limit=35)

    existing_list = [
        it.value.get("data", "")
        for it in items
        if isinstance(it.value, dict)
    ]

    existing_memory = "\n".join(existing_list) if existing_list else "(empty)"
    existing_set = set(existing_list)

    # 🔹 2. Prepare last messages context
    human_msg = state["messages"][-1].content

    # 🔹 3. Build parser + prompt
    parser = PydanticOutputParser(pydantic_object=MemoryDecision)

    prompt_template = PromptTemplate(
    template="""
Return ONLY valid JSON.
Schema:
{format_instructions}
{{
  "need_to_remember": boolean,
  "new_memories": [
    {{
      "memory": string,
      "memory_type": string
    }}
  ]
}}
CURRENT USER DETAILS:
{existing_memory}
LAST CHAT:
{human_msg}
---
Goal
Decide whether the LAST CHAT contains NEW long-term information that should be added to memory.
Only store information that is likely to improve future conversations.
---
Decision Rules
Set "need_to_remember" = true ONLY when the chat contains NEW and REUSABLE information such as:
- Preferences
- Long-term goals
- Identity/background
- Skills or expertise
- Ongoing projects
- Stable habits
- Constraints
- Learning style
- Career or education information
- Persistent interests
- Important decisions that affect future interactions
Set "need_to_remember" = false when the message is:
- Casual conversation
- Greetings
- Small talk
- Temporary situations
- Emotional reactions
- One-time questions
- Follow-up questions
- Clarifications
- Explanations
- Requests for information
- Problem-solving discussions that do not reveal reusable user information
Memory Extraction Rules
Extract ONLY information that:
1. Is NEW (not already present in CURRENT USER DETAILS)
2. Is useful in future conversations
3. Is likely to remain true for weeks or months
4. Helps personalize future responses
Do NOT extract:
- Normal conversation content
- Temporary plans
- Single-session context
- Questions asked by the user
- Facts about topics being discussed
- Assistant responses
- Speculation or assumptions
- Information already present in CURRENT USER DETAILS
Each memory must be:
- Atomic (one fact only)
- Short
- Self-contained
- Normalized
- Written from the user's perspective
---
Allowed memory_type values
personal
habit
interests
goals
skills
dislikes
preferences
learning_style
projects
tools
constraints
knowledge_level
career
education
behavior
decisions
context
health
Use EXACTLY ONE memory_type per memory.
---
Consistency Rules
- If need_to_remember = false, new_memories MUST be []
- If need_to_remember = true, new_memories MUST contain at least one item
- Never create duplicate memories
- Never combine multiple facts into one memory
- Never invent information
---
Example
{{
  "need_to_remember": true,
  "new_memories": [
    {{
      "memory": "User prefers short responses",
      "memory_type": "preferences"
    }}
  ]
}}
""",
    input_variables=["existing_memory", "human_msg"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)
    chain = prompt_template | llm_summarizer | parser
    decision = await chain.ainvoke(
        {
            "existing_memory": existing_memory,
            "human_msg": human_msg
        }
    )
    if not decision.need_to_remember:
        return state
    
    new_unique_memories = [
    mem
    for mem in decision.new_memories
    if mem.memory.strip() and mem.memory.strip() not in existing_set
]
    if not new_unique_memories:
        return state
    dt= get_current_date()
    async with AsyncPostgresStore.from_conn_string(
        DB_POSTGRESSTORE_PATH,
        index={
            "embed": postgres_embed,
            "dims": 1024,
            "text_fields": ["data"]   # tells the store which field to embed
        }) as put_store:



        await put_store.setup()
        await asyncio.gather(
            *[
                put_store.aput(
                    namespace,
                    str(uuid4()),
                    {
                        "data": mem.memory,
                        "type": mem.memory_type,
                        "date": dt[0],
                        "time": dt[1],
                        },
                        )
                        for mem in new_unique_memories
            ]
            )

    # 🔹 6. Return state unchanged + trace
    return {"trace": update_trace(state, "Remember Node")}
#-----------------Retriever-nodes------------------------------------------------


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
- file_name like 'ml_notes.pdf'

- file_hash if available.
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

class InfoFetcher_node(BaseModel):
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



async def get_document_summaries(
    conn: AsyncConnection,
    user_id: str,
    thread_id: str,
) -> str:
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT file_hash, summary
            FROM uploaded_documents
            WHERE user_id = %s
              AND thread_id = %s
            ORDER BY uploaded_at
            """,
            (user_id, thread_id),
        )

        rows = await cur.fetchall()

    return "\n\n".join(
        f"Source: {file_hash}\nSummary: {summary or 'No summary available'}"
        for file_hash, summary in rows
    )

async def retrieval_info_fetcher_node(state: ChatBotState,config: RunnableConfig):
    """
    Node that inspects the conversation and populates state with
    InfoFetcher_node parameters (user_query, memory plans, doc plans).
    Routing decisions are left entirely to downstream edges/nodes.
    """

    thread_id = config['configurable']['thread_id']
    user_id = state['user_details']['user_id']

    message = state["messages"][-1] if  isinstance(state["messages"][-1],HumanMessage) else "no"
    try:
        async with await AsyncConnection.connect(DB_POSTGRESSTORE_PATH) as conn:
            source = await get_document_summaries(
                conn=conn,
                user_id=user_id,
                thread_id=thread_id
            )
    except:
        source = "(empty)"

    if message =="no":
        return {
        "retrieval_details":{
            "user_msg": "",
            "rag_details":[],
            "user_memories": []
        }}
    system_message  = state["system_messages"][0]
    retrieval_type = state['retrieval_type']
    retrieval_scope = "\n".join([
        "- retrieve from uploaded documents" if "uploaded_documents" in retrieval_type else "",
        "- retrieve from user memories" if "user_memories" in retrieval_type else ""
        ]).strip()
    # ── LLM call ─────────────────────────────────────────────────────────────
    parser = PydanticOutputParser(pydantic_object=InfoFetcher_node)

    prompt = PromptTemplate(
    template="""
You are a retrieval planning system.

Retrieval has already been confirmed as necessary.
Your ONLY job is to fill in the retrieval parameters as precisely as possible.

━━━ Retrieval scope (already decided) ────────────────────────────────────
{retrieval_scope}

Only populate the fields that match the scope above.
Ignore and leave null any field not listed in the scope.

━━━ user_query ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Copy the EXACT latest user message here, character-for-character.
Never rewrite, clean, or summarise it.

━━━ user_memories_retrieval_details ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Populate ONLY if "user_memories" is listed in the retrieval scope above.

Covers: preferences · habits · goals · projects · skills ·
        learning style · conversational continuity · past decisions

Rules:
- One FetchUserMemoryDetails object per memory category
- search_query must be optimised for retrieval — NOT a copy of user_query
- Choose the single most relevant filter_by_type per object
- Prefer multiple focused objects over one broad query

━━━ uploaded_documents_retrieval_details ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Populate ONLY if "uploaded_documents" is listed in the retrieval scope above.

Covers: uploaded files · PDFs · URLs · notes · vector-stored knowledge bases

Rules:
- One FetchUploadedDocsDetails object per retrieval strategy
- search_query must be a concise semantic query — NOT a copy of user_query
- Use "similarity" for precise, targeted lookup
- Use "mmr" when broader or more diverse context is needed
- Set filter_by_source only when a specific file or source is clearly implied

━━━ Query optimisation rules (both fields) ──────────────────────────────
- search_query is NEVER a copy of user_query
- Remove conversational filler ("can you", "please", "I want to know")
- Preserve key entities, concepts, and intent
- Keep queries short and retrieval-focused

━━━ System message ────────────────────────────────────────────────────────
{system_message}


━━━ available sources ────────────────────────────────────────────────────────
{source}

━━━ Latest user query ─────────────────────────────────────────────────────
{query}

{format_instructions}
""",

    input_variables=["source","retrieval_scope", "system_message", "query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain    = prompt | llm_summarizer | parser
    result: InfoFetcher_node = await chain.ainvoke({
        "retrieval_scope":retrieval_scope,
        "source":source,
        "system_message": system_message.content,
        "query":message.content,
    })

    return {
        "retrieval_details":{
            "user_msg":result.user_query or message.content,
            "rag_details":result.uploaded_documents_retrieval_details or [],
            "user_memories":result.user_memories_retrieval_details or []
        },
        "trace":update_trace(state,"Retrieval Decision stage 1")}

#  docs
async def rag_result(vector_store:VectorStore,user_msg,search_query,top_k,search_type,source):# This function performs a retrieval-augmented generation (RAG) process by querying the vector store with the provided search query and parameters. It constructs the search parameters based on whether a specific source filter is applied, retrieves relevant documents using the retriever, and then uses the summarization LLM to analyze the retrieved content in relation to the user's query. The prompt instructs the model to determine if the retrieved information is relevant and useful for answering the query, and to provide a concise answer based solely on that information, or to indicate if no relevant information is available.
    if source:
        search_kwargs={
            "k":top_k or 6,
            "filter":{
                "file_hash": source
        }
    }
    else:
        search_kwargs={
            "k":top_k or 6
            }
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    result_docs = await  retriever.ainvoke(search_query)
    retrieved_content = "\n".join([doc.page_content for doc in result_docs]) if result_docs else "(no content retrieved)"
    prompt = f"""
You are given:
1. A user query
2. Retrieved content from uploaded documents

Your task:
- Analyze whether the retrieved content is actually relevant to the user's query.
- If the retrieved content contains useful information related to the query, provide a clear and concise answer using only the retrieved information.
- If the retrieved content is unrelated, weakly related, noisy, or does not contain enough useful information to answer the query, then return exactly:

"No information related to your query is available in the uploaded documents."

User Query:
{user_msg}          ← original user message

Retrieval Query Used:
{search_query}       ← what was searched

Retrieved Content:
{retrieved_content}

Answer:
"""
    res = await llm_summarizer.ainvoke(prompt)
    return res.content

async def retriever_node(state: ChatBotState,config:RunnableConfig):

    if not state['retrieval_details']['rag_details']:
        return {
            "retriever_context_message":None
        }

    thread_id = config['configurable']['thread_id']
    user_id = state['user_details']['user_id']
    query_list = state['retrieval_details']['rag_details']
    user_msg = state['retrieval_details']['user_msg']

    vector_store = load_vectorstore(user_id,thread_id)
    if not vector_store:
        return {
            "retriever_context_message": SystemMessage(
                content="No uploaded documents found for this conversation.")
                }
    query_list_result = []
    for query in query_list:
        result_rag = await rag_result(
            vector_store=vector_store,
            user_msg=user_msg,
            search_type=query.retrieval_mode,
            search_query=query.search_query,
            source=query.filter_by_source,
            top_k=query.num_docs
            )

        query_list_result.append(f"""Query for RAG: {query.search_query}\nRAG response: {result_rag} \n source: {query.filter_by_source  if query.filter_by_source else "Not mentioned"}""")

    total_results = "\n\n".join(query_list_result)
    retriever_context_message = SystemMessage(content=f"""

User Original Query:
{user_msg}

RAG Retrieval Results:
{total_results}

Final Consolidated Answer:
""")
    return {
        "retrieval_details":{
            "rag_details" : []
        },
        "retriever_context_message" : retriever_context_message
    }

# user memories
async def retrieve_user_memory_node(state: ChatBotState, store: BaseStore): # This node is responsible for retrieving relevant user memories from the vector store based on the current user query and the specified retrieval details. It constructs a consolidated system message that includes the retrieved memories, which can then be used by the LLM to provide more informed and personalized responses to the user's query. The node ensures that only relevant and helpful memories are included in the system message, while ignoring any unrelated or non-useful information.
    query_list = state["retrieval_details"]['user_memories']
    if not query_list:
        return {
            "user_details": {
                "user_memory": None
            }
        }
    main_query = state['retrieval_details']['user_msg']
    user_id = state['user_details']['user_id']
    namespace = ("user", user_id, "details")
    query_results = []
    for query in query_list:
        fetched_result = await get_BasicMemories(
            namespace=namespace,
            filter_by_type=query.filter_by_type,
            search_query=query.search_query,
            num_docs=query.num_docs,
            store=store
        )
        query_results.append(fetched_result)
    result = "\n\n".join(query_results)
    result_message = f"""
These are newly retrieved memories related to the current user query.

User query:
"{main_query}"

Retrieved memories:
{result}

Use these memories only if they are helpful and relevant for answering the user query.
If the retrieved memories seem unrelated or not useful, ignore them.
"""
    if state['user_details']["user_memory"] is not None:
        old_memories = state["user_details"]['user_memory']
        new_memories = result_message
        prompt = f"""Your work is to summarize the old and new memories system message.
        rule:
        - max 650 characters,Only most important information allowed for 650+ characters. else hard limit is 650 characters
        - generate A clear Summary.
        - remove duplicate information.
        - remove unrelated information.

        old message : {old_memories}


        new message : {new_memories}

        summary output:
        """
        response = await llm_summarizer.ainvoke(prompt)
        return {
            "user_details": {
            "user_memory": response.content
            }
            }

    if len(result_message) > 800:
        prompt = f"""
You are a memory summarizer.

You will receive a list of retrieved user memories.
Your job is to compress them into a single dense summary
that will be injected into a chat assistant's context.

Rules:
- Maximum 500 characters — hard limit, never exceed it
- Every word must earn its place — no filler, no repetition
- Preserve concrete facts: names, numbers, tools, preferences, decisions
- Drop vague or low-signal memories (e.g. "user likes things to be simple")
- Write in third person, present tense ("User prefers...", "User is building...")
- Output ONLY the summary — no labels, no preamble, no explanation

Memories:
{result_message}
"""
        response = await llm_summarizer.ainvoke(prompt)
        memory_message = response.content
    
    else:
        memory_message = result_message

    update_system_msg = state["system_messages"]+[SystemMessage(content=memory_message)]
    return {
        "user_details": {
            "user_memory": memory_message
        }
    }

class Evaluate_Retrieval_Content(BaseModel):
    retriever_score:float = Field(
        description=(
            "Score the quality and relevance of the retrieved RAG documents "
            "for answering the user's query. Higher scores indicate the "
            "retrieved documents are highly relevant, complete, accurate, and "
            "sufficient to answer the query. Lower scores indicate the "
            "documents are irrelevant, incomplete, outdated, or missing key information. "
            "Return a value between 0 and 100."
            "0-20: Irrelevant or unusable. "
            "21-40: Mostly irrelevant with little useful information. "
            "41-60: Partially relevant but incomplete. "
            "61-80: Relevant and useful with minor gaps. "
            "81-100: Highly relevant, accurate, and sufficient."
        )
        ,default=0.0
        ,ge=0.0,
        le=100.0
        )
    memories_score:float = Field(description=(
            "Score the quality and usefulness of the retrieved memories for "
            "answering the user's query. Higher scores indicate the memories "
            "are highly relevant, accurate, personalized, and helpful. "
            "Lower scores indicate the memories are irrelevant, outdated, "
            "contradictory, or provide little value for the current query. "
            "Return a value between 0 and 100."
            "0-20: Irrelevant or unusable. "
            "21-40: Mostly irrelevant with little useful information. "
            "41-60: Partially relevant but incomplete. "
            "61-80: Relevant and useful with minor gaps. "
            "81-100: Highly relevant, accurate, and sufficient."
        ),default=0.0,ge=0.0,le=100.0)
async def retrieval_evaluation_node(state: ChatBotState):
    retriever_content = f"retriever content :\n{state['retriever_context_message'].content}" if state['retriever_context_message'] else "no retriever is used so keep retriever_score : 0.0"
    user_memories_content = f"fetched User Memories:\n{state['system_messages'][1].content}" if len(state['system_messages']) >1 else "Memories fetcher is not used so keep memories_score : 0.0"
    parser = PydanticOutputParser(pydantic_object=Evaluate_Retrieval_Content)
    prompt = PromptTemplate(
        template="""You are an expert retrieval evaluator.
{format_instructions}
Your task is to evaluate the quality of the retrieved RAG documents and retrieved memories for answering the user's query.

{retriever_content}

{user_memories_content}

Evaluate both retrieval sources independently.

Scoring Guidelines:

RAG Score (0-100):

* 0-20: Documents are irrelevant or unusable.
* 21-40: Mostly irrelevant with little useful information.
* 41-60: Partially relevant but incomplete.
* 61-80: Relevant and useful with minor gaps.
* 81-100: Highly relevant, accurate, complete, and sufficient to answer the query.

Memories Score (0-100):

* 0-20: Memories are irrelevant or incorrect.
* 21-40: Memories have little value for the current query.
* 41-60: Some memories are useful but important context is missing.
* 61-80: Memories are relevant and provide meaningful personalization.
* 81-100: Memories are highly relevant, accurate, and significantly improve the response quality.

Consider:

1. Relevance to the user's query.
2. Completeness of the information.
3. Accuracy and consistency.
4. Whether the retrieved information would help generate a better answer.
5. Whether important information appears to be missing.

Return only the structured output.
""",
    input_variables=['retriever_content','user_memories_content'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    result:Evaluate_Retrieval_Content = await chain.ainvoke(
        {
            "retriever_content":retriever_content,
            "user_memories_content":user_memories_content
        }
    )
    retriever_score = result.retriever_score if result.retriever_score else 0.0
    memories_score = result.memories_score if result.memories_score else 0.0
    return {
        "retrieval_score":{
            "retriever_score":retriever_score,
            "memories_score":memories_score
        }

    }
