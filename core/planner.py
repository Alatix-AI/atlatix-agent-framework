# agentforge/core/planner.py
import json
import math
import asyncio
import time
import re
from json import JSONDecodeError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterable

"""
Advanced multi-step planner (CrewAI-style)
- PlanGraph / PlanNode data structures
- Multi-candidate generation (Tree-of-Thought)
- Candidate scoring / ranking
- Self-refinement of nodes
- Step-by-step execution planning & tool-chaining
- Dead-end detection and replanning
- Function-calling aware when LLM adapter supports it

Usage (high-level):
    planner = AdvancedPlanner(llm_adapter)
    action = await planner.plan(task, history, context=context, tools=tool_registry.list_schemas())

Returned action is normalized:
    {"type":"tool","name":..., "input":{...}}
or
    {"type":"final","output":"..."}
"""

# ----------------------------
# Basic types
# ----------------------------
ScoreRationale = Tuple[float, str]

# ----------------------------
# PlanNode & PlanGraph
# ----------------------------
@dataclass
class PlanNode:
    """A single node in the plan graph representing a candidate next-step."""
    id: int
    candidate: Dict[str, Any]             # raw candidate dict from LLM
    normalized: Dict[str, Any]            # normalized (type/tool/final) candidate
    parent_id: Optional[int] = None
    depth: int = 0
    score: float = 0.0
    rationale: str = ""
    refined: bool = False
    executed: bool = False                # whether node has been executed in simulation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanGraph:
    """Container for nodes (a forest/tree)."""
    nodes: Dict[int, PlanNode] = field(default_factory=dict)
    root_ids: List[int] = field(default_factory=list)
    next_id: int = 1

    def add_node(self, candidate: Dict[str, Any], normalized: Dict[str, Any], parent_id: Optional[int] = None, depth: int = 0, meta: Dict[str, Any] = None) -> PlanNode:
        nid = self.next_id
        self.next_id += 1
        node = PlanNode(id=nid, candidate=candidate, normalized=normalized, parent_id=parent_id, depth=depth, metadata=meta or {})
        self.nodes[nid] = node
        if parent_id is None:
            self.root_ids.append(nid)
        return node

    def children_of(self, node_id: int) -> List[PlanNode]:
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def all_roots(self) -> List[PlanNode]:
        return [self.nodes[rid] for rid in self.root_ids]

    def path_to_root(self, node_id: int) -> List[PlanNode]:
        path = []
        cur = self.nodes.get(node_id)
        while cur:
            path.append(cur)
            cur = self.nodes.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(path))


# ----------------------------
# Planner implementation
# ----------------------------
class AdvancedPlanner:
    """
    CrewAI-style multi-step planner.

    Key parameters:
      - n_candidates: how many candidate next-steps to generate per expansion
      - max_depth: how deep the planner will expand
      - candidate_temperature: sampling temperature for candidate generation
      - eval_temperature: temperature for scoring/evaluation prompts
      - prune_threshold: score threshold below which branches are pruned
      - max_nodes: safety cap on number of nodes created
    """

    def __init__(
        self,
        llm_adapter,
        n_candidates: int = 3,
        max_depth: int = 3,
        candidate_temperature: Optional[float] = None,  
        eval_temperature: Optional[float] = None,        
        max_retries: int = 2,
        prune_threshold: float = 0.15,
        max_nodes: int = 50,
    ):
        self.llm = llm_adapter
        self.n_candidates = n_candidates
        self.max_depth = max_depth
        self.candidate_temperature = candidate_temperature if candidate_temperature is not None else 0.9
        self.eval_temperature = eval_temperature if eval_temperature is not None else 0.0
        self.max_retries = max_retries
        self.prune_threshold = prune_threshold
        self.max_nodes = max_nodes

    # ----------------------------
    # Public entry
    # ----------------------------
    async def plan(
        self,
        task: str,
        history: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a next action (tool call or final) using multi-step planning.
        The planner constructs a PlanGraph of candidate action paths, expands up to max_depth,
        scores candidates and returns the best normalized action.

        context: {"semantic": [...], "episodic": [...]} as produced by MemorySystem.retrieve_context()
        tools: dict name -> schema (if provided, function-calling path may be used)
        """
        if context is None:
            context = {"semantic": [], "episodic": []}

        base_prompt = self._build_base_prompt(task, history, context)

        # Build initial graph by generating root candidates
        graph = PlanGraph()
        root_candidates = await self._generate_candidates(base_prompt, tools=tools, n=self.n_candidates)

        if not root_candidates:
            # fallback single-step
            return await self._single_step_fallback(base_prompt, tools=tools)

        # add root nodes
        for cand in root_candidates:
            norm = self._normalize_candidate(cand)
            graph.add_node(candidate=cand, normalized=norm, parent_id=None, depth=0)

        # expand graph (breadth-first up to max_depth)
        await self._expand_graph(graph, task, history, context, tools)

        # evaluate and pick best leaf node
        best_node = await self._select_best_node(graph, task, history, context)

        if not best_node:
            # fallback
            return await self._single_step_fallback(base_prompt, tools=tools)

        # Return the first actionable step from that node's path (top-most tool/final)
        chosen_norm = await self._choose_action_from_path(graph, best_node)
        return chosen_norm

    # ----------------------------
    # Graph expansion
    # ----------------------------
    async def _expand_graph(self, graph: PlanGraph, task: str, history: List[Dict[str, Any]], context: Dict[str, Any], tools: Optional[Dict[str, Any]]):
        """
        Expand nodes breadth-first. For each leaf node, simulate its effect minimally and produce child candidates.
        Stop when:
          - reached max_depth
          - nodes exceed max_nodes
          - no promising candidates
        """
        frontier = list(graph.all_roots())
        while frontier:
            next_frontier = []
            for node in frontier:
                if node.depth >= self.max_depth:
                    continue
                if len(graph.nodes) >= self.max_nodes:
                    return
                # small dead-end heuristic: skip if node scored extremely low in previous eval
                if node.score is not None and node.score < self.prune_threshold:
                    continue

                # Build expansion prompt using path context
                path = graph.path_to_root(node.id)
                expand_prompt = self._build_expansion_prompt(task, history, context, path, tools)

                # generate children candidates
                children = await self._generate_candidates(expand_prompt, tools=tools, n=self.n_candidates)
                if not children:
                    # try refine on parent and retry once
                    refined_parent = await self._maybe_refine_candidate(self._format_path_block(path), path[-1].candidate if path else {})
                    if refined_parent:
                        path[-1].candidate = refined_parent
                        path[-1].normalized = self._normalize_candidate(refined_parent)
                        children = await self._generate_candidates(expand_prompt, tools=tools, n=self.n_candidates)

                for cand in children:
                    norm = self._normalize_candidate(cand)
                    child = graph.add_node(candidate=cand, normalized=norm, parent_id=node.id, depth=node.depth + 1)
                    # quick evaluation for pruning
                    score, rationale = await self._score_candidate(task, history, context, cand)
                    child.score = score
                    child.rationale = rationale
                    # dead-end detection: very low score or repeating action in path
                    if self._is_dead_end(graph, child):
                        child.score = min(child.score, self.prune_threshold * 0.5)
                    # only add promising children to next frontier
                    if child.score >= self.prune_threshold:
                        next_frontier.append(child)
                    # safety stop
                    if len(graph.nodes) >= self.max_nodes:
                        break
                if len(graph.nodes) >= self.max_nodes:
                    break
            frontier = next_frontier

    # ----------------------------
    # Candidate generation (single-shot list of candidates)
    # ----------------------------
    async def _generate_candidates(self, prompt: str, tools: Optional[Dict[str, Any]] = None, n: int = 3) -> List[Dict[str, Any]]:
        """
        Ask the model to return a JSON array of candidate objects.
        Prefer plain generation; function-calling path is reserved for the final execution step (single-shot).
        """
        tools_block = self._format_tools_block(tools) if tools else "(no tools provided)"
        gen_prompt = f"""
You are an AI planner. Produce exactly {n} distinct candidate next steps for the agent.
Each candidate must be a JSON object and the full response must be a JSON array (list).
Allowed candidate forms:
 - Call a tool:
   {{"type":"tool", "name":"<tool-name>", "input": {{ ... }}}}
 - Final answer:
   {{"type":"final", "output":"..."}}

Context:
{prompt}

Available tools:
{tools_block}

Return EXACTLY {n} JSON candidate objects inside a JSON array. No extra text.
"""
        # sample for diversity
        raw = await self.llm.generate(gen_prompt, temperature=self.candidate_temperature)
        text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
        parsed = self._try_parse_json(text)
        if parsed and isinstance(parsed, list):
            return [p for p in parsed if isinstance(p, dict)][:n]

        # retry with deterministic prompt
        raw2 = await self.llm.generate(gen_prompt + "\nIMPORTANT: Output must be valid JSON array, nothing else.", temperature=0.0)
        text2 = raw2 if isinstance(raw2, str) else getattr(raw2, "text", str(raw2))
        parsed2 = self._try_parse_json(text2)
        if parsed2 and isinstance(parsed2, list):
            return [p for p in parsed2 if isinstance(p, dict)][:n]
        return []

    # ----------------------------
    # Candidate scoring 
    # ----------------------------
    async def _score_candidate(self, task: str, history: List[Dict[str, Any]], context: Dict[str, Any], candidate: Dict[str, Any]) -> ScoreRationale:
        """
        Ask the model to score candidate between 0.0 and 1.0 and return (score, rationale).
        Uses eval_temperature for deterministic scoring when appropriate.
        """
        cand_text = json.dumps(candidate, ensure_ascii=False)
        hist = "\n".join([f"- action: {h['action']} | result: {str(h['result'])[:200]}" for h in history])

        score_prompt = f"""
You are an evaluator. Rate how good the following candidate is for the TASK and give a numeric score 0.0-1.0 (higher is better).
TASK:
{task}

HISTORY:
{hist if hist else '(no history)'}

CONTEXT:
{json.dumps(context)}

CANDIDATE:
{cand_text}

Provide output as JSON exactly in the form:
{{"score": <float_between_0_and_1>, "rationale": "<one-sentence explanation>"}}
"""
        raw = await self.llm.generate(score_prompt, temperature=self.eval_temperature)
        text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
        parsed = self._try_parse_json(text)
        if parsed and isinstance(parsed, dict) and "score" in parsed:
            try:
                s = float(parsed.get("score", 0.0))
                r = str(parsed.get("rationale", ""))[:1000]
                s = max(0.0, min(1.0, s))
                return s, r
            except Exception:
                pass

        # fallback: look for a numeric token
        try:
            import re
            m = re.search(r"([01](?:\.\d+)?)", text)
            if m:
                s = float(m.group(1))
                s = max(0.0, min(1.0, s))
                return s, "parsed fallback"
        except Exception:
            pass
        return 0.0, "failed to parse score"

    # ----------------------------
    # Node refinement
    # ----------------------------
    async def _maybe_refine_candidate(self, context_text: str, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Ask LLM to fix/normalize a candidate. Returns dict or None.
        """
        cand_text = json.dumps(candidate, ensure_ascii=False)
        refine_prompt = f"""
Candidate to check:
{cand_text}

Context / instructions:
{context_text}

If there's any minor error (invalid JSON, missing fields, wrong param names) correct the candidate and return the corrected JSON object.
Return ONLY the JSON object (no extra text). If candidate is fine, return it unchanged.
"""
        raw = await self.llm.generate(refine_prompt, temperature=0.0)
        text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
        parsed = self._try_parse_json(text)
        if parsed and isinstance(parsed, dict):
            return parsed
        return None

    # ----------------------------
    # Selection & extraction
    # ----------------------------
    async def _select_best_node(self, graph: PlanGraph, task: str, history: List[Dict[str, Any]], context: Dict[str, Any]) -> Optional[PlanNode]:
        """
        Choose the best leaf node by score. If tie or ambiguity, ask model to compare top-k candidate paths.
        """
        # gather leaves
        leaves = [n for n in graph.nodes.values() if not graph.children_of(n.id)]
        if not leaves:
            leaves = list(graph.nodes.values())

        # sort by score
        leaves.sort(key=lambda x: x.score or 0.0, reverse=True)
        topk = leaves[: min(5, len(leaves))]

        if len(topk) == 0:
            return None
        if len(topk) == 1:
            return topk[0]

        # if top scores are close, ask LLM to compare
        if topk[0].score - topk[1].score < 0.05:
            compare_prompt = self._build_compare_prompt(task, history, context, topk)
            raw = await self.llm.generate(compare_prompt, temperature=self.eval_temperature)
            text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
            parsed = self._try_parse_json(text)
            # Expecting {"pick": <index>} or simple "pick N"
            if parsed and isinstance(parsed, dict) and "pick" in parsed:
                try:
                    idx = int(parsed.get("pick"))
                    if 0 <= idx < len(topk):
                        return topk[idx]
                except Exception:
                    pass
            # fallback: return topk[0]
        return topk[0]

    async def _choose_action_from_path(self, graph: PlanGraph, node: PlanNode) -> Dict[str, Any]:
        """
        Given a selected node (leaf), return the first actionable step from root->leaf path.
        If the path contains intermediate tooling, return the earliest tool call.
        """
        path = graph.path_to_root(node.id)
        # find first actionable element (tool or final) - usually root is actionable
        for p in path:
            if p.normalized.get("type") in ("tool", "final"):
                return p.normalized
        # last resort
        return path[-1].normalized if path else {"type": "final", "output": ""}

    # ----------------------------
    # Utility: single-step fallback
    # ----------------------------
    async def _single_step_fallback(self, prompt: str, tools: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ask the LLM for a single-step decision (compatibility fallback).
        """
        # if function-calling available and tools provided, try one-shot function-calling
        if tools and hasattr(self.llm, "generate_with_functions"):
            functions = list(tools.values()) if isinstance(tools, dict) else tools
            try:
                raw = await self.llm.generate_with_functions(prompt, functions=functions)
                # normalize function-calling response similar to earlier code
                if isinstance(raw, dict):
                    choices = raw.get("choices") or []
                    if choices:
                        msg = choices[0].get("message") or {}
                        func_call = msg.get("function_call") or None
                        if func_call:
                            fname = func_call.get("name")
                            args_raw = func_call.get("arguments") or "{}"
                            try:
                                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                            except Exception:
                                args = {"__raw": args_raw}
                            return {"type": "tool", "name": fname, "input": args}
                        # else final text
                        content = msg.get("content") or ""
                        if content:
                            parsed = self._try_parse_json(content)
                            if parsed and isinstance(parsed, dict) and parsed.get("type"):
                                return self._normalize_candidate(parsed)
                            return {"type": "final", "output": content}
            except NotImplementedError:
                pass
            except Exception:
                pass

        # normal text path
        for attempt in range(self.max_retries + 1):
            raw = await self.llm.generate(prompt)
            text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
            parsed = self._try_parse_json(text)
            if parsed:
                if isinstance(parsed, dict) and parsed.get("type") == "tool":
                    args = parsed.get("arguments") or parsed.get("input") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"__raw": args}
                    return {"type": "tool", "name": parsed.get("name"), "input": args}
                if isinstance(parsed, dict) and parsed.get("type") == "final":
                    return {"type": "final", "output": parsed.get("output")}
            # refine ask
            refine = "Your previous response was not valid. Return exactly one JSON object matching the schema."
            raw2 = await self.llm.generate(prompt + "\n" + refine, temperature=0.0)
            text2 = raw2 if isinstance(raw2, str) else getattr(raw2, "text", str(raw2))
            parsed2 = self._try_parse_json(text2)
            if parsed2 and isinstance(parsed2, dict):
                return self._normalize_candidate(parsed2)
        return {"type": "final", "output": "[Planner failed]"}

    # ----------------------------
    # Dead-end detection heuristics
    # ----------------------------
    def _is_dead_end(self, graph: PlanGraph, node: PlanNode) -> bool:
        """
        Heuristics:
         - Repeated tool call cycles in path (same tool + similar input)
         - Node normalized type is tool but tool not in available registry (can't be executed) — handled elsewhere
         - Very small score (already applied)
        """
        path = graph.path_to_root(node.id)
        seen = set()
        repeats = 0
        for p in path:
            if p.normalized.get("type") == "tool":
                key = (p.normalized.get("name"), json_dumps_stable(p.normalized.get("input")))
                if key in seen:
                    repeats += 1
                seen.add(key)
        # if repeated the same tool twice in path → likely loop
        if repeats >= 1:
            return True
        return False

    # ----------------------------
    # Prompts & formatting helpers
    # ----------------------------
    def _build_base_prompt(self, task: str, history: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        hist = "\n".join([f"- action: {h['action']} | result: {str(h['result'])[:300]}" for h in history])
        semantic_block = self._format_semantic(context.get("semantic", []))
        episodic_block = self._format_episodic(context.get("episodic", []))
        base = f"""
You are an advanced multi-step planner for an autonomous agent.

TASK:
{task}

RELEVANT SEMANTIC MEMORY (top matches):
{semantic_block}

RECENT EPISODIC:
{episodic_block}

HISTORY:
{hist if hist else '(no previous steps)'}
"""
        return base

    def _build_expansion_prompt(self, task: str, history: List[Dict[str, Any]], context: Dict[str, Any], path: List[PlanNode], tools: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a prompt describing the current partial path so the LLM can generate next-step candidates that chain from it.
        """
        path_block = self._format_path_block(path)
        base = self._build_base_prompt(task, history, context)
        tools_block = self._format_tools_block(tools) if tools else "(no tools provided)"
        return f"""{base}

CURRENT PLAN PATH:
{path_block}

Please propose next steps that follow naturally from the above partial plan.
Available tools:
{tools_block}
"""
    def _build_compare_prompt(self, task: str, history: List[Dict[str, Any]], context: Dict[str, Any], candidates: List[PlanNode]) -> str:
        items = []
        for i, c in enumerate(candidates):
            items.append(f"INDEX: {i}\nSCORE: {c.score}\nCANDIDATE: {json.dumps(c.candidate)}\nRATIONALE: {c.rationale}\n")
        items_block = "\n\n".join(items)
        return f"""
You are an evaluator comparing multiple FULL candidate plan paths. TASK: {task}
HISTORY: {json.dumps(history)}
CONTEXT: {json.dumps(context)}

Candidates:
{items_block}

Based on the task and context, choose the best candidate by returning JSON: {{ "pick": <index_of_best_candidate> }}
"""
    def _format_path_block(self, path: Iterable[PlanNode]) -> str:
        lines = []
        for p in path:
            lines.append(f"- depth:{p.depth} type:{p.normalized.get('type')} name:{p.normalized.get('name')} input:{json_dumps_stable(p.normalized.get('input'))} score:{p.score:.3f} rationale:{p.rationale}")
        return "\n".join(lines) if lines else "(empty path)"

    def _format_semantic(self, semantic_hits: List[Dict[str, Any]], max_chars: int = 300) -> str:
        if not semantic_hits:
            return "(no semantic hits)"
        lines = []
        for h in semantic_hits:
            lines.append(f"- score:{h.get('score'):.4f} text: {str(h.get('text',''))[:max_chars].replace('\\n',' ')}")
        return "\n".join(lines)

    def _format_episodic(self, episodic: List[Dict[str, Any]], max_chars: int = 500) -> str:
        if not episodic:
            return "(no episodic)"
        lines = []
        for e in episodic:
            lines.append(f"- action:{e.get('action')} result:{str(e.get('result'))[:max_chars].replace('\\n',' ')}")
        return "\n".join(lines)

    def _format_tools_block(self, tools: Optional[Dict[str, Any]]) -> str:
        if not tools:
            return "(no tools available)"
        lines = []
        if isinstance(tools, dict):
            items = tools.items()
        else:
            items = [(t.get("name"), t) for t in tools]
        for name, schema in items:
            desc = schema.get("description", "")
            params = schema.get("parameters", {})
            lines.append(f"- {name}: {desc} | params: {json.dumps(params)}")
        return "\n".join(lines)

    # ----------------------------
    # JSON parse / normalize utilities
    # ----------------------------

    def _try_parse_json(self, text: str) -> Optional[Any]:
        if not text or not isinstance(text, str):
            return None

        # 1. Triple backticks + optional "json" tag temizle
        text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*$", "", text, flags=re.IGNORECASE)

        # 2. Önce tam metni dene
        try:
            return json.loads(text)
        except JSONDecodeError:
            pass

        # 3. En dıştaki geçerli JSON objesini/arraysini bul (sağdan sola güvenli parse)
        #    Örn: "Explain: {"x":1} and {"y":2}" → sadece {"y":2} döner (en sonuncu)
        #    Ama genellikle model sadece bir JSON üretir.

        # JSON objesi için güvenli parse
        depth = 0
        start = None
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except JSONDecodeError:
                        continue  # geçersizse bir sonraki }'ye bak

        # JSON array için güvenli parse
        depth = 0
        start = None
        for i, char in enumerate(text):
            if char == '[':
                if depth == 0:
                    start = i
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except JSONDecodeError:
                        continue

        return None
    

    def _normalize_candidate(self, cand: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize various candidate shapes into unified schema:
          - {"type":"tool","name":..,"input":{..}}
          - {"type":"final","output": "..."}
        """
        if not isinstance(cand, dict):
            return {"type": "final", "output": str(cand)}
        t = cand.get("type") or cand.get("action") or cand.get("kind")
        if t == "tool" or (not t and cand.get("name") and (cand.get("input") or cand.get("arguments"))):
            name = cand.get("name", None)  # safe fallback below
        # robust extraction:
        t = cand.get("type") or cand.get("action") or None
        if t == "tool" or (not t and cand.get("name")):
            name = cand.get("name") or cand.get("tool") or cand.get("function") or cand.get("func")
            inp = cand.get("input") or cand.get("arguments") or cand.get("args") or {}
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except Exception:
                    inp = {"__raw": inp}
            return {"type": "tool", "name": name, "input": inp}
        if t == "final":
            return {"type": "final", "output": cand.get("output") or cand.get("text") or ""}
        # fallback heuristics
        if cand.get("name"):
            inp = cand.get("input") or cand.get("arguments") or {}
            return {"type": "tool", "name": cand.get("name"), "input": inp}
        return {"type": "final", "output": str(cand)}

# ----------------------------
# Small helpers
# ----------------------------
def json_dumps_stable(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"

def json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)
