 # agentforge/cli.py
import argparse
import asyncio
import os
import sys
from typing import List

from agentforge import Agent, tools
from agentforge.core.tools import ToolRegistry


def list_builtin_tools() -> dict:
    return {
        "http": tools.http,
    }


def load_tools(tool_names: List[str]) -> List:
    builtin = list_builtin_tools()
    selected = []
    for name in tool_names:
        if name in builtin:
            selected.append(builtin[name]())
        else:
            print(f"⚠️ Warning: Built-in tool '{name}' not found. Skipping.", file=sys.stderr)
    return selected


async def run_agent_async(task: str, model: str, tool_names: List[str]):
    if not task:
        print("❌ Error: --task is required.", file=sys.stderr)
        sys.exit(1)

    selected_tools = load_tools(tool_names)

    try:
        agent = Agent(model=model, tools=selected_tools)

        print(f"🧠 Agent started with model: {model}")
        if selected_tools:
            print(f"🧰 Enabled tools: {', '.join(tool_names)}")
        print(f"🎯 Task: {task}\n")
        print("⏳ Thinking...\n")

        # Use streaming for better UX
        async for event in agent.run_stream(task):
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] == "tool":
                print(f"\n🔧 **Tool Used**: `{event['name']}`")
                print(f"   Input: {event['input']}")
                if event["result"]["success"]:
                    print(f"   Output: {event['result']['output']}")
                else:
                    print(f"   ❌ Error: {event['result']['output']['error']}")
                print("\n⏳ Continuing...\n")
            elif event["type"] == "final":
                if not event.get("text", "").strip():
                    print("\n💡 Final output was empty.")
                # final text already printed via tokens if streaming
                pass

    except KeyboardInterrupt:
        print("\n\n🛑 Agent interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="AgentForge: Build and run autonomous AI agents."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The task you want the agent to perform (e.g., 'Summarize https://example.com')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai:gpt-4o-mini",
        help="LLM model to use (e.g., 'openai:gpt-4o-mini', 'ollama:llama3')"
    )
    parser.add_argument(
        "--tools",
        type=str,
        default="calc,http",
        help="Comma-separated list of built-in tools (e.g., 'calc,http'). Available: calc, http"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (return final answer only)"
    )

    args = parser.parse_args()

    # Validate env (basic)
    if args.model.startswith("openai:") and not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is required for OpenAI models.", file=sys.stderr)
        sys.exit(1)

    tool_list = [t.strip() for t in args.tools.split(",")] if args.tools else []

    if args.no_stream:
        agent = Agent(model=args.model, tools=load_tools(tool_list))
        result = asyncio.run(agent.run(args.task))
        print(result)
    else:
        asyncio.run(run_agent_async(args.task, args.model, tool_list))


if __name__ == "__main__":
    main()