"""Tool adapter utilities for released benchmark runs."""
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ToolDefinition(BaseModel):

    name: str
    summary: str
    parameters: List[Dict]
    returns: List[Dict]
    exceptions: List[Dict] = []


class InjectionConfig(BaseModel):

    enabled: bool = False
    target_tool: Optional[str] = None
    position: str = "append"
    content: Optional[str] = None


class ToolAdapter:


    def __init__(self, toolemu_defs_path: Optional[str] = None, simulator_llm=None):


        self.execution_stats = {}


        self.toolemu_defs_path = toolemu_defs_path
        self.simulator_llm = simulator_llm

    def get_available_tools(self) -> List[str]:


        from . import virtual_tools
        return getattr(virtual_tools, '__all__', [])

    def get_openai_tools(self, tool_names: List[str]) -> List:


        try:
            from . import virtual_tools
            tools = []
            for name in tool_names:
                if hasattr(virtual_tools, name):
                    tools.append(getattr(virtual_tools, name))
            return tools
        except ImportError:
            return []

    def record_tool_call(self, tool_name: str):


        if tool_name not in self.execution_stats:
            self.execution_stats[tool_name] = 0
        self.execution_stats[tool_name] += 1

    def get_tool_stats(self) -> Dict[str, int]:


        return self.execution_stats.copy()

    def reset_stats(self):

        self.execution_stats.clear()
