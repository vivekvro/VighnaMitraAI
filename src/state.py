from pydantic import Field
from typing import TypedDict,Annotated,List,Optional,Any,Literal
from langchain_core.messages import BaseMessage,SystemMessage
from langgraph.graph.message import add_messages




class BaseChatState(TypedDict):
    system_messages:List[SystemMessage]
    messages:Annotated[List[BaseMessage],add_messages]

class SummaryState(TypedDict):
    summary_content :Annotated[str,"Updated summary combining previous + new conversation chunk"]=None
    summary_end_index: int

class UserDetails(TypedDict):
    user_id:str
    user_memory:Optional[str]
class Retrieval_schema(TypedDict):
    user_msg:str
    rag_details: List[Any]
    user_memories: List[Any]

class Retrieval_scores(TypedDict):
    retriever_score:float
    memories_score:float
class ChatBotState(BaseChatState):
    summary:SummaryState
    retrieval_details:Optional[Retrieval_schema]
    retrieval_type: List[Literal["uploaded_documents","user_memories"]]
    retrieval_score:Retrieval_scores
    retriever_context_message:Optional[SystemMessage]
    user_details:UserDetails
    trace:List[str]