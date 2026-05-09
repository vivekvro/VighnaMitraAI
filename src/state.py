from pydantic import Field
from typing import Literal,TypedDict,Annotated,Dict,List,Any,Tuple,Optional
from langchain_core.messages import BaseMessage,SystemMessage
from langgraph.graph.message import add_messages
from langchain_community.vectorstores import VectorStore
from src.chatbots.node_condtions import FetchUploadedDocsDetails,FetchUserMemoryDetails



class BaseChatState(TypedDict):
    system_message:SystemMessage
    messages:Annotated[List[BaseMessage],add_messages]

class SummaryState(TypedDict):
    summary_content :Annotated[str,"Updated summary combining previous + new conversation chunk"]=None
    summary_end_index: int

class UserDetails(TypedDict):
    user_id:str
    user_memory:Optional[str]
class Retrieval_schema(TypedDict):
    retrieval_type: str | list
    uploaded_documents: List[Any]
    user_memories: List[Any]

class ChatBotState(BaseChatState):
    summary:SummaryState
    retrieval_details:Optional[Retrieval_schema]
    user_details:UserDetails
    trace:List[str]=Field(
        default_factory=list,
        description="in this add used tools and nodes.")