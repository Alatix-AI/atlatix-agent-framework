from .core.agent import Agent
from .core.tools import tool
import subprocess
import zipfile
import shutil
from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any, Literal
from tavily import TavilyClient
import tempfile
import time

# Initialize Tavily client for web search
tavily_client = TavilyClient(api_key="....")
# ----------------- Tools -----------------
    
@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance","science","technology","economy"] = "general",
    include_raw_content: bool = False,
)-> List[Dict[str, Any]]:
    
    """
    Tool to perform an internet search using the Tavily API.

    This tool allows the agent to gather information from the web
    based on a query and a specified topic. It returns a list of
    search results, optionally including the raw content of the
    webpages.

    Args:
        query (str): The search query or keywords to look up on the web.
        max_results (int, optional): Maximum number of search results to return. 
                                     Defaults to 5.
        topic (Literal["general", "news", "finance", "science", "technology", "economy"], optional): 
            Category of the search to prioritize relevant content. Defaults to "general".
        include_raw_content (bool, optional): If True, include the full raw content of the results; 
                                             otherwise, only metadata is returned. Defaults to False.

    Returns:
        List[Dict[str, Any]]: A list of search results from Tavily, with each item containing
                              relevant information such as title, URL, snippet, and optionally raw content.
    """

    result1 = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return result1

@tool
def code_executor(
    image: str,
    cmds: List[str],
    mounts: Dict[str, str] = None,
    host_workspace: str = None,
    container_workdir: str = "/workspace",
    timeout: int = 60,
    allow_network: bool = False,
) -> Dict[str, Any]:
    
    """
    Executes a sequence of shell commands inside a Docker container.

    This tool allows safe and isolated execution of code or scripts
    using a specified Docker image. It supports mounting host directories,
    custom working directories, timeout handling, and optional network access.

    Args:
        image (str): The Docker image to use for execution (e.g., "python:3.11-slim").
        cmds (List[str]): A list of shell commands to run inside the container.
        mounts (Dict[str, str], optional): Dictionary mapping host paths to container paths
                                           for volume mounting. Defaults to None.
        host_workspace (str, optional): Path on the host machine to use as workspace.
                                        If None, a temporary directory is created. Defaults to None.
        container_workdir (str, optional): Working directory inside the container. Defaults to "/workspace".
        timeout (int, optional): Maximum execution time in seconds before terminating the process. Defaults to 60.
        allow_network (bool, optional): Whether to allow network access inside the container.
                                        Defaults to False (safe default).

    Returns:
        Dict[str, Any]: A dictionary containing execution results:
            - stdout (str): Standard output from the container.
            - stderr (str): Standard error output.
            - exit_code (int): Exit code of the executed commands.
            - runtime_s (float): Execution time in seconds.
            - files (List[str]): List of files created in the host workspace (relative paths).
            - host_workspace (str): Path to the host workspace used for execution.

    Notes:
        - Ensures that the host workspace is always mounted to the container.
        - Normalizes Windows paths for Docker volume mounting.
        - Safely handles subprocess timeouts and captures output.
    """

    if host_workspace is None:
        host_workspace = tempfile.mkdtemp(prefix="mini_manus_ws_")
    # Ensure mounts include host_workspace -> container_workdir
    mounts = dict(mounts or {})
    if host_workspace not in mounts:
        mounts[host_workspace] = container_workdir

    docker_cmd = ["docker", "run", "--rm", "--memory", "512m", "--cpus", "1"]
    if not allow_network:
        docker_cmd += ["--network", "none"]

    # Normalize Windows backslashes -> forward slashes for docker -v on some setups
    def _norm(p: str) -> str:
        return p.replace("\\", "/")

    for host, cont in mounts.items():
        docker_cmd += ["-v", f"{_norm(host)}:{cont}"]

    docker_cmd += ["-w", container_workdir, image]
    joined = " && ".join(cmds) if cmds else "echo 'No commands provided'"
    docker_cmd += ["sh", "-lc", joined]

    start = time.time()
    try:
        proc = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout)
        runtime = time.time() - start

        # Gather files from the host workspace (NOT container path)
        files = []
        try:
            for p in Path(host_workspace).rglob("*"):
                if p.is_file():
                    files.append(str(p.relative_to(host_workspace)))
        except Exception:
            files = []

        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "runtime_s": round(runtime, 3),
            "files": files,
            "host_workspace": host_workspace,
        }
    except subprocess.TimeoutExpired as te:
        return {
            "stdout": te.stdout or "",
            "stderr": (te.stderr or "") + f"\n[Timed out after {timeout}s]",
            "exit_code": -1,
            "runtime_s": round(time.time() - start, 3),
            "files": [],
            "host_workspace": host_workspace,
        }
@tool
def save_files(manifest_files: List[Dict[str,str]], workspace: str = None) -> str:
    
    """
    Saves a list of files to a host workspace directory.

    This tool creates the specified files with their content on the host system.
    Each file is defined by a dictionary containing a relative path and content.
    If no workspace path is provided, a temporary directory is created automatically.

    Args:
        manifest_files (List[Dict[str, str]]): A list of file descriptors, 
            where each descriptor is a dictionary with:
            - "path" (str): Relative file path (e.g., "app.py" or "src/module.py").
            - "content" (str): The content to write into the file.
        workspace (str, optional): Path to the host directory where files should be saved.
                                   If None, a temporary directory is created. Defaults to None.

    Returns:
        str: The path to the host workspace directory where the files were saved.

    Notes:
        - Automatically creates parent directories if they do not exist.
        - Overwrites files if they already exist at the same path.
        - Useful for preparing workspaces for code execution in sandboxed environments.
    """

    if workspace is None:
        workspace = tempfile.mkdtemp(prefix="mini_manus_ws_")
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    for f in manifest_files:
        p = ws / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f["content"], encoding="utf-8")
    return str(ws)

# 2) List files in a workspace (relative)
@tool
def list_workspace_files(workspace: str) -> List[str]:

    """
    Recursively list all files in a given workspace directory.

    This tool traverses the workspace directory and collects all file paths,
    returning them relative to the workspace root. It is useful for inspecting 
    the contents of a workspace, packaging artifacts, or tracking generated files.

    Args:
        workspace (str): Path to the workspace directory to list.

    Returns:
        List[str]: A list of file paths relative to the workspace root.

    Notes:
        - Only files are included; directories themselves are ignored.
        - If the workspace path is invalid or an error occurs during traversal,
          an empty list is returned.
        - Paths are returned as strings using forward slashes.
    """

    files = []
    try:
        for p in Path(workspace).rglob("*"):
            if p.is_file():
                files.append(str(p.relative_to(workspace)))
    except Exception:
        pass
    return files

# 3) Package artifact (zip) and return path
@tool
def package_artifact(workspace: str, out_dir: str = None) -> str:

    """
    Package the contents of a workspace directory into a ZIP archive.

    This tool collects all files within a given workspace and compresses 
    them into a single ZIP file, which can be used as an artifact for 
    deployment, sharing, or backup purposes.

    Args:
        workspace (str): Path to the workspace directory to package.
        out_dir (str, optional): Directory to save the generated ZIP file. 
            If None, a temporary directory will be created.

    Returns:
        str: Absolute file path of the created ZIP archive.

    Notes:
        - Only files are included in the ZIP archive; directories themselves 
          are not stored.
        - The ZIP filename is automatically generated using a UUID to ensure 
          uniqueness.
        - If `out_dir` does not exist, it will be created.
        - Useful for packaging code, data, or other artifacts generated 
          during automated workflows.
    """

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="mini_manus_artifacts_")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    zip_name = Path(out_dir) / f"artifact_{uuid4().hex}.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for p in Path(workspace).rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(workspace))
    return str(zip_name)

# 4) Cleanup workspace
@tool
def cleanup_workspace(workspace: str, keep: bool = False) -> None:

    """
    Safely removes a workspace directory and all its contents.

    This tool is used to clean up temporary directories created during 
    code execution, testing, or file manipulation. It ensures that the 
    workspace is deleted unless explicitly preserved.

    Args:
        workspace (str): Path to the workspace directory to delete.
        keep (bool, optional): If True, the workspace will not be deleted.
            Defaults to False.

    Returns:
        None

    Notes:
        - Any errors during deletion (e.g., non-existent directory, permission issues) 
          are silently ignored.
        - Use `keep=True` to preserve the workspace, for example, when artifacts 
          need to be inspected after execution.
        - Intended for host-side cleanup of temporary directories used in containerized 
          or local code execution workflows.
    """

    if keep:
        return
    try:
        shutil.rmtree(workspace)
    except Exception:
        pass

# 5) Run a manifest end-to-end using your code_executor (uses Docker image + run_commands)
@tool
def run_manifest(manifest: Dict[str, Any], base_image: str = "python:3.11-slim", timeout: int = 120, keep_workspace: bool = False) -> Dict[str, Any]:
    
    """
    Executes a manifest of files and commands inside a Docker container and optionally packages the workspace.

    This tool automates the process of:
    1. Saving provided files to a host workspace.
    2. Installing dependencies (if a `requirements.txt` is present or if `install_libs` is specified).
    3. Running commands and optional test commands inside a Docker container.
       - Commands referencing workspace files are automatically adjusted to point to the container workspace.
    4. Collecting outputs, listing files, and optionally packaging the workspace into a ZIP artifact.
    5. Cleaning up the workspace unless `keep_workspace=True`.

    Args:
        manifest (Dict[str, Any]): A dictionary describing the manifest, with the following keys:
            - "files" (List[Dict[str,str]]): List of files to save, each with "path" and "content".
            - "run_commands" (List[str], optional): Commands to execute inside the container.
            - "test_command" (str, optional): A command for testing/verifying the execution.
            - "install_libs" (List[str], optional): A list of Python packages to install dynamically
              (e.g., ["crewai", "transformers"]). Installed before any run/test commands.
        base_image (str, optional): Docker image to use for execution. Defaults to "python:3.11-slim".
        timeout (int, optional): Maximum time in seconds for container execution. Defaults to 120.
        keep_workspace (bool, optional): If True, preserves the host workspace after execution. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing execution results and metadata:
            - "stdout" (str): Standard output from the execution.
            - "stderr" (str): Standard error from the execution.
            - "exit_code" (int): Exit code of the executed commands.
            - "runtime_s" (float): Total runtime in seconds.
            - "files" (List[str]): List of files present in the workspace after execution.
            - "artifact" (str or None): Path to a ZIP file of the workspace, if packaging succeeded.
            - "workspace" (str): Path to the host workspace.

    Notes:
        - If `requirements.txt` exists, dependencies are installed automatically inside the container.
        - If `install_libs` is provided, those packages are installed dynamically via pip.
        - Commands that reference workspace files are automatically adjusted to point to the container workspace.
        - Network access is enabled briefly during dependency installation.
        - Commands are executed sequentially inside the container.
        - Workspace cleanup is automatic unless `keep_workspace=True`.
        - Useful for safely running and testing code in isolated, reproducible environments.
    """

    files = manifest.get("files", [])
    run_cmds = manifest.get("run_commands", [])
    test_cmd = manifest.get("test_command")
    install_libs = manifest.get("install_libs", [])   # 👈 NEW
    host_workspace = save_files(files)  # this returns a host path

    # Map host workspace -> container path
    mounts = {host_workspace: "/workspace"}

    # Pre-install step if requirements.txt exists
    install_cmds = []
    if install_libs:
        # install arbitrary packages inside container
        libs = " ".join(install_libs)
        install_cmds.append(f"pip install {libs}")

    if (Path(host_workspace) / "requirements.txt").exists():
        install_cmds.append("pip install -r requirements.txt")

    #NEW 
    def fix_file_paths(cmds: List[str]) -> List[str]:
        fixed = []
        for c in cmds:
            parts = c.split()
            if parts[0] == "python" and len(parts) > 1:
                parts[1] = f"/workspace/{parts[1]}"
            fixed.append(" ".join(parts))
        return fixed
    

    # Build the full command sequence (run installs first if present)

    run_cmds = fix_file_paths(run_cmds)
    if test_cmd:
        test_cmd = fix_file_paths([test_cmd])[0]

    # Build full command list
    cmds = install_cmds + [f"cd /workspace && {c}" for c in run_cmds]
    if test_cmd:
        cmds.append(f"cd /workspace && {test_cmd}")

    if not cmds:
        cmds = ["cd /workspace && echo 'No commands provided'"]


    # If we're installing requirements, allow network briefly (set allow_network=True)
    allow_network = bool(install_cmds)

    exec_res = code_executor(
        image=base_image,
        cmds=cmds,
        mounts=mounts,
        host_workspace=host_workspace,
        container_workdir="/workspace",
        timeout=timeout,
        allow_network=allow_network,
    )

    # gather host-side file list (relative)
    files_list = list_workspace_files(host_workspace)

    # package artifact (optional)
    artifact = None
    try:
        artifact = package_artifact(host_workspace)
    except Exception:
        artifact = None

    result = {
        "stdout": exec_res.get("stdout", ""),
        "stderr": exec_res.get("stderr", ""),
        "exit_code": exec_res.get("exit_code", 1),
        "runtime_s": exec_res.get("runtime_s", None),
        "files": files_list,
        "artifact": artifact,
        "workspace": host_workspace,
    }

    # decide whether to cleanup workspace
    cleanup_workspace(host_workspace, keep=keep_workspace)
    return result

# ----------------- Agent Initialization -----------------
Codeagent = Agent(
    model="huggingface:Qwen/Qwen3-Coder-30B-A3B-Instruct:nebius",
    api_key="hf_xxx",
    max_tokens=4048,
    tools=[
        internet_search,
        code_executor,
        save_files,
        list_workspace_files,
        package_artifact,
        cleanup_workspace,
        run_manifest,
    ], 
    max_steps=5,
    #base_url="http://example...v1",
    temperature=0.6,
    persistent=True  # use persistent semantic memory
)

# ----------------- Example Run -----------------
async def main():

    query = "Search the internet for the latest advancements about Aİ Agent and build data agent using crewai library. Save the code and any necessary files."
    result = await Codeagent.run(query)
    print("Final Result:\n", result)



# -----------------------------------------------
# Multi-Agent setup can be added here if needed
# -----------------------------------------------
Codeagent = Agent(
    model="huggingface:Qwen/Qwen3-Coder-30B-A3B-Instruct:nebius",
    api_key="hf_xxx",
    max_tokens=4048,
    tools=[read_file,...],
    max_steps=5,
    temperature=0.6,
    persistent=True,
    name="Coding Agent"
)

Researchagent = Agent(
    model="openai:gpt-3.5-turbo",
    api_key="sk-xxx",
    max_tokens=4048,
    tools=[write_file,...],
    max_steps=2,
    temperature=0.6,
    persistent=True,
    name="Researcher Agent"
)

# Ana orchestrator
Main = SubAgentOrchestrator(
    agents=[Codeagent, Researchagent],
    workspace_dir=".agentforge/shared_workspace"
)

async def main():
    query = "Search the internet for the latest advancements about AI Agent and build data agent using crewai library. Save the code and any necessary files."
    result = await Main.run(query)
    print("Final Result:\n", result)

# Çalıştır
import asyncio
asyncio.run(main())
