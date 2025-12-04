# agentforge/core/tools.py
from typing import Callable, Any, Dict, Optional
import inspect
import json
import asyncio


class ToolExecutionError(Exception):
    pass


class Tool:
    """
    Unified Tool interface:
    - name
    - callable function (sync/async)
    - JSON schema for LLM
    - safe execution with standardized ToolResult
    """

    def __init__(self, name: str, func: Callable, description: str = ""):
        self.name = name
        self.func = func
        self.description = description
        self.schema = self._build_schema(func, name, description)

    # ------------------------------------------------------------
    # Build JSON schema from Python function signature
    # ------------------------------------------------------------
    def _build_schema(self, func: Callable, name: str, description: str) -> Dict[str, Any]:
        sig = inspect.signature(func)
        params = {}
        required = []

        for pname, p in sig.parameters.items():
            # Tür tahmini
            param_type = self._guess_json_type(p.annotation)

            # Zorunlu mu?
            is_required = p.default == inspect.Parameter.empty

            params[pname] = {
                "type": param_type,
                "description": f"Parameter {pname}"
            }
            if is_required:
                required.append(pname)

        return {
            "name": name,
            "description": description or f"Tool {name}",
            "parameters": {
                "type": "object",
                "properties": params,
                "required": required, 
            },
        }

    def _guess_json_type(self, annot):
        if annot in [int, float]:
            return "number"
        if annot == bool:
            return "boolean"
        if annot == dict:
            return "object"
        if annot == list:
            return "array"
        return "string"

    # ------------------------------------------------------------
    # Safe execution wrapper → unified ToolResult
    # ------------------------------------------------------------
    async def run(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async-safe execution wrapper for async or sync tool functions.
        """
        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**inp)
            else:
                # Senkron fonksiyonu async context'te çalıştır
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: self.func(**inp))

            # normalize output
            if isinstance(result, str):
                output = {"text": result}
            elif isinstance(result, dict):
                output = result
            else:
                output = {"value": result}

            return {
                "type": "tool",
                "name": self.name,
                "success": True,
                "output": output,
                "raw": result
            }

        except Exception as e:
            return {
                "type": "tool",
                "name": self.name,
                "success": False,
                "output": {"error": str(e)},
                "raw": None
            }

    def __repr__(self):
        return f"<Tool name={self.name}>"


# ------------------------------------------------------------
# Tool Registry
# ------------------------------------------------------------
class ToolRegistry:
    """
    - Register tools
    - Lookup by name
    - Expose schemas for LLM
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def add(self, tool: Tool):
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_schemas(self) -> Dict[str, Any]:
        return {name: t.schema for name, t in self.tools.items()}


# ------------------------------------------------------------
# Built-in tools
# ------------------------------------------------------------
def http():
    def _run(url: str) -> str:
        import requests
        resp = requests.get(url, timeout=10)
        return resp.text

    return Tool("http_fetch", _run, description="Fetch content from a URL")


def python_function(func: Callable, name: str = None, description: str = ""):
    return Tool(name or func.__name__, func, description)


def tool(func=None, *, name=None, description=None):
    """
    Decorator to turn a function into a Tool.
    
    Usage:
        @tool
        def calc(x: int, y: int) -> int:
            '''Add two numbers.'''
            return x + y

        # or with custom name/description
        @tool(name="adder", description="Adds two integers")
        def my_func(a, b): ...
    """
    def decorator(f):
        nonlocal name, description
        tool_name = name or f.__name__
        tool_desc = description or (f.__doc__ or "").strip()
        return Tool(tool_name, f, tool_desc)
    
    if func is None:
        return decorator
    else:
        return decorator(func)
 

 
 


 
