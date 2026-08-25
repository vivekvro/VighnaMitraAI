from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from pathlib import Path
import json
import asyncio
class ToolConfigLocal(BaseModel):
    command: str = Field(
        default="uv",
        description="Executable used to run the tool (e.g., uv, python, node)"
    )
    args: List[str] = Field(
        default_factory=list,
        description="List of command-line arguments passed to the command"
    )
    transport: Literal["stdio", "http", "websocket"] = Field(
        default="stdio",
        description="Communication method between client and tool"
    )
    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables required for the tool execution"
    )
    cwd: Optional[Path] = Field(
        default=None,
        description="Working directory where the command will be executed"
    )
async def load_config():
    with open("src/configs/mcpServers_config.json","r") as f:
        data = json.load(f)
    return data
async def update_config_local(servername:str,configs:ToolConfigLocal):
    data = await load_config()
    data[servername] = configs.model_dump()
    with open("src/configs/mcpServers_config.json","w") as f:
        json.dump(data,f,indent=4)
    return"Config updated!."

from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, Optional, Literal
class ToolConfigRemote(BaseModel):
    url: HttpUrl = Field(
        description="Base URL of the MCP server"
    )
    transport: Literal["http", "websocket"] = Field(
        default="http",
        description="Transport protocol used to communicate with MCP server"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional headers (e.g., Authorization)"
    )
    timeout: int = Field(
        default=30,
        description="Connection/request timeout in seconds"
    )
    reconnect: bool = Field(
        default=True,
        description="Whether to auto-reconnect (useful for websocket transport)"
    )
    auth_token: Optional[str] = Field(
        default=None,
        description="Optional API/Bearer token"
    )




