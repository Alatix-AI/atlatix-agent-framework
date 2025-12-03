# agentforge/core/memory.py
from __future__ import annotations
from typing import List, Any, Dict, Tuple, Optional
import numpy as np
import asyncio
import time
import json
import faiss
import pickle
from pathlib import Path


# ------------------------------------------------------------
# SIMPLE VECTOR STORE (in-memory)
# - stores np arrays and metadata in lists
# - minimal API: add / search_by_vector / delete_indices
# ------------------------------------------------------------


class SimpleVectorStore:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vec: np.ndarray, meta: Dict[str, Any]):
        if vec.shape[0] != self.dim:
            raise ValueError(f"Vector dim mismatch: expected {self.dim}, got {vec.shape[0]}")
        self.vectors.append(vec)
        self.metadata.append(meta)

    def search_by_vector(self, query_vec: np.ndarray, k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        if len(self.vectors) == 0:
            return []
        mat = np.stack(self.vectors)  # [N, D]
        mat_norm = np.linalg.norm(mat, axis=1) + 1e-8
        q_norm = np.linalg.norm(query_vec) + 1e-8
        sims = (mat @ query_vec) / (mat_norm * q_norm)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.metadata[i]) for i in idx]

    def delete_indices(self, indices: List[int]):
        """Delete metadata/vectors at given indices (indices assumed sorted ascending)."""
        if not indices:
            return
        keep_vecs = []
        keep_meta = []
        to_delete = set(indices)
        for i, (v, m) in enumerate(zip(self.vectors, self.metadata)):
            if i not in to_delete:
                keep_vecs.append(v)
                keep_meta.append(m)
        self.vectors = keep_vecs
        self.metadata = keep_meta

    def __len__(self):
        return len(self.vectors)

# ------------------------------------------------------------
# FAISS VECTOR STORE (with async snapshot persistence)
# - stores np arrays and metadata in lists
# - uses FAISS for efficient similarity search
# - async snapshot persistence to disk
# ------------------------------------------------------------

class FAISSVectorStore:
    def __init__(self, dim: int = 768, index_path: Optional[str] = None):
        self.dim = dim
        self.index_path = Path(index_path) if index_path else None
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        
        # FAISS index (L2 mesafe)
        self.faiss_index = faiss.IndexFlatL2(dim)
        
        # Snapshot & persistence
        self._dirty_count = 0
        self._max_dirty_before_save = 10
        self._lock = asyncio.Lock()

        # Diskten yükle
        if self.index_path and self.index_path.exists():
            self._load_from_disk()

    def add(self, vec: np.ndarray, meta: Dict[str, Any]):
        if vec.shape[0] != self.dim:
            raise ValueError(f"Vector dim mismatch: expected {self.dim}, got {vec.shape[0]}")
        self.vectors.append(vec.copy())  # orijinal vektörü sakla
        self.metadata.append(meta)
        self.faiss_index.add(vec.reshape(1, -1).astype('float32'))
        self._dirty_count += 1
        asyncio.create_task(self._maybe_save_async())

    def search_by_vector(self, query_vec: np.ndarray, k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        if len(self.vectors) == 0:
            return []
        query = query_vec.reshape(1, -1).astype('float32')
        distances, indices = self.faiss_index.search(query, min(k, len(self.vectors)))
        results = []
        for d, i in zip(distances[0], indices[0]):
            if i != -1:
                # L2 mesafesi → negatif skor olarak döndür (SimpleVectorStore ile uyumlu)
                results.append((float(-d), self.metadata[i]))
        return results

    def delete_indices(self, indices: List[int]):
        if not indices or not self.vectors:
            return
        indices = sorted(set(indices))  # uniq ve sıralı
        keep_mask = [True] * len(self.vectors)
        for i in indices:
            if 0 <= i < len(keep_mask):
                keep_mask[i] = False

        new_vectors = [v for v, keep in zip(self.vectors, keep_mask) if keep]
        new_meta = [m for m, keep in zip(self.metadata, keep_mask) if keep]

        self.vectors = new_vectors
        self.metadata = new_meta
        # FAISS index'i yeniden inşa et
        self.faiss_index = faiss.IndexFlatL2(self.dim)
        if new_vectors:
            mat = np.stack(new_vectors).astype('float32')
            self.faiss_index.add(mat)
        self._dirty_count += len(indices)
        asyncio.create_task(self._maybe_save_async())

    def __len__(self):
        return len(self.vectors)

    # ----------------------------
    # Async Snapshot Persistence
    # ----------------------------
    async def _maybe_save_async(self):
        async with self._lock:
            if self._dirty_count >= self._max_dirty_before_save and self.index_path:
                await asyncio.get_event_loop().run_in_executor(None, self._save_to_disk)
                self._dirty_count = 0

    def _save_to_disk(self):
        if not self.index_path:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Vektörleri ve metadata'yı pickle ile kaydet
            with open(self.index_path.with_suffix(".vectors.pkl"), "wb") as f:
                pickle.dump(self.vectors, f)
            with open(self.index_path.with_suffix(".meta.pkl"), "wb") as f:
                pickle.dump(self.metadata, f)
            # FAISS index'i ayrı kaydet
            faiss.write_index(self.faiss_index, str(self.index_path.with_suffix(".faiss")))
        except Exception as e:
            print(f"Warning: FAISS snapshot failed: {e}")

    def _load_from_disk(self):
        vectors_path = self.index_path.with_suffix(".vectors.pkl")
        meta_path = self.index_path.with_suffix(".meta.pkl")
        faiss_path = self.index_path.with_suffix(".faiss")
        
        if vectors_path.exists() and meta_path.exists() and faiss_path.exists():
            try:
                with open(vectors_path, "rb") as f:
                    self.vectors = pickle.load(f)
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                self.faiss_index = faiss.read_index(str(faiss_path))
                # Uyumluluk kontrolü
                if len(self.vectors) != len(self.metadata):
                    raise ValueError("Vector and metadata length mismatch")
            except Exception as e:
                print(f"Warning: FAISS load failed, starting fresh: {e}")
                self.vectors = []
                self.metadata = []
                self.faiss_index = faiss.IndexFlatL2(self.dim)


# ------------------------------------------------------------
# EPISODIC MEMORY (advanced)
# - stores raw events
# - chunk-based summarization using LLM
# - provides recent events + recent summaries for context
# ------------------------------------------------------------
class EpisodicMemoryAdvanced:
    def __init__(self, llm_adapter, max_items: int = 200, chunk_size: int = 20, summary_retention: int = 10):
        self.max_items = max_items
        self.memory: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self.llm = llm_adapter
        self.chunk_size = max(5, int(chunk_size))
        self.summary_retention = int(summary_retention)
        # list of dicts: {"summary": str, "from_idx": int, "to_idx": int, "ts": float}
        self.summaries: List[Dict[str, Any]] = []

    async def add(self, item: Dict[str, Any]):
        """
        Add an episodic event: item should be a dict like {"action":..., "result":...}
        Every chunk_size additions we create an LLM-generated summary for the last chunk.
        """
        async with self._lock:
            self.memory.append(item)
            if len(self.memory) > self.max_items:
                # drop earliest items to maintain max_items
                drop = len(self.memory) - self.max_items
                self.memory = self.memory[drop:]
                # adjust summaries indices
                for s in self.summaries:
                    s["from_idx"] = max(0, s["from_idx"] - drop)
                    s["to_idx"] = max(0, s["to_idx"] - drop)
                # drop outdated summaries
                self.summaries = [s for s in self.summaries if s["to_idx"] >= s["from_idx"]]

            # if we reached a new chunk boundary -> summarize last chunk (fire-and-forget)
            if len(self.memory) % self.chunk_size == 0:
                # create a task but don't await here — summarization runs in background
                asyncio.create_task(self._summarize_last_chunk())

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.memory)

    def search_last(self, k: int = 3) -> List[Dict[str, Any]]:
        return list(self.memory[-k:])

    async def _summarize_last_chunk(self):
        """Summarize the most recent chunk_size items and append to summaries."""
        async with self._lock:
            if len(self.memory) == 0:
                return
            to_idx = len(self.memory) - 1
            from_idx = max(0, to_idx - self.chunk_size + 1)
            chunk = self.memory[from_idx: to_idx + 1]
            # build text for summarization
            texts = []
            for e in chunk:
                act = e.get("action")
                res = e.get("result")
                texts.append(f"Action: {json.dumps(act, ensure_ascii=False)}\nResult: {json.dumps(res, ensure_ascii=False)}")
            chunk_text = "\n\n".join(texts)
        # call LLM outside lock (to avoid blocking)
        try:
            prompt = (
                "Summarize the following sequence of agent actions and observations into a concise bullet-point summary.\n\n"
                f"{chunk_text}\n\n"
                "Provide a short summary (1-3 sentences) capturing the key outcomes and any important facts."
            )
            raw = await self.llm.generate(prompt)
            summary_text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
        except Exception as e:
            summary_text = f"[summary failed: {str(e)}]"

        async with self._lock:
            # store summary metadata
            self.summaries.append({
                "summary": summary_text,
                "from_idx": from_idx,
                "to_idx": to_idx,
                "ts": time.time()
            })
            # keep only last N summaries
            if len(self.summaries) > self.summary_retention:
                self.summaries = self.summaries[-self.summary_retention:]

    def get_recent_summaries(self, n: int = 3) -> List[Dict[str, Any]]:
        return list(self.summaries[-n:])


# ------------------------------------------------------------
# SEMANTIC MEMORY (advanced)
# - Maintains vector store + metadata (importance, last_access, freq)
# - On retrieval it boosts metadata (frequency/last_access)
# - Periodic forgetting via importance threshold or max size
# ------------------------------------------------------------
class SemanticMemoryAdvanced:
    def __init__(
        self,
        llm_adapter,
        dim: int = 768,
        max_items: int = 5000,
        forget_threshold: float = 0.05,
        importance_decay: float = 0.995,
        persistent: bool = False,      # ← EKLENDİ
        index_path: Optional[str] = None  # ← EKLENDİ
    ):
        self.llm = llm_adapter
        if persistent:
            if index_path is None:
                index_path = f".agentforge/semantic_{int(time.time())}.index"
            self.index = FAISSVectorStore(dim=dim, index_path=index_path)
        else:
            self.index = SimpleVectorStore(dim=dim)
        self.max_items = int(max_items)
        self.forget_threshold = float(forget_threshold)
        self.importance_decay = float(importance_decay)
        self._lock = asyncio.Lock()


    async def add(self, text: str, metadata: Dict[str, Any] = None):
        """
        Add semantic item. Metadata will be extended with importance/freq/last_access.
        """
        metadata = dict(metadata or {})
        metadata.setdefault("text", text)
        metadata.setdefault("importance", float(metadata.get("importance", 1.0)))
        metadata.setdefault("frequency", int(metadata.get("frequency", 0)))
        metadata.setdefault("last_access", float(metadata.get("last_access", time.time())))
        # embed
        try:
            vectors = await self.llm.embed([text])
            if not vectors or not isinstance(vectors[0], (list, tuple, np.ndarray)):
                raise RuntimeError("Embed returned unexpected format")
            vec = np.array(vectors[0], dtype=float)

            async with self._lock:
                if len(self.index.vectors) == 0 and self.index.dim == 768:
                    # İlk kez embedding geliyorsa, dimension’ı ayarla
                    self.index.dim = vec.shape[0]
                elif vec.shape[0] != self.index.dim:
                    raise ValueError(f"Embedding dimension mismatch: expected {self.index.dim}, got {vec.shape[0]}")

                self.index.add(vec, metadata)
                # enforce max_items (simple FIFO pruning of lowest importance)
                await self._prune_if_needed()
        except NotImplementedError:
            return 
        
    async def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        try:
            vectors = await self.llm.embed([query])
        except NotImplementedError:
            return []
        
        if not vectors:
            return []
        qvec = np.array(vectors[0], dtype=float)
        async with self._lock:
            hits = self.index.search_by_vector(qvec, k=k)
            results = []
            for score, meta in hits:
                # update metadata inplace (frequency + last_access)
                meta["frequency"] = meta.get("frequency", 0) + 1
                meta["last_access"] = time.time()
                # optionally bump importance slightly
                meta["importance"] = min(10.0, meta.get("importance", 1.0) + 0.02)
                results.append({"score": score, **meta})
            # after retrieval, possibly decay importances
            self._decay_importance_locked()
            # maybe prune
            await self._prune_if_needed()
            return results

    def _decay_importance_locked(self):
        # apply decay to all metadata entries; needs to be called under lock
        for meta in self.index.metadata:
            meta["importance"] = float(meta.get("importance", 1.0)) * self.importance_decay

    async def _prune_if_needed(self):
        """Prune items based on importance threshold and max_items; needs to be called under lock."""
        if len(self.index) <= self.max_items:
            # but still trim items with extremely low importance
            low_idxs = [i for i, m in enumerate(self.index.metadata) if m.get("importance", 1.0) < self.forget_threshold]
            if low_idxs and len(self.index) - len(low_idxs) >= 1:
                # delete low importance entries
                self.index.delete_indices(low_idxs)
                return

        # if too many items, keep top by importance
        metas = self.index.metadata
        importances = np.array([m.get("importance", 0.0) for m in metas])
        if importances.size == 0:
            return
        # get indices of top max_items
        keep_idx = list(np.argsort(-importances)[: self.max_items])
        keep_idx_sorted = sorted(keep_idx)
        self.index.delete_indices([i for i in range(len(self.index)) if i not in keep_idx_sorted])


    async def cluster_and_summarize(self, cluster_size: int = 10, summary_max_chars: int = 1000) -> List[Dict[str, Any]]:
        """
        Simple greedy clustering based on cosine similarity to produce cluster summaries.
        Returns list of cluster summaries (metadata blobs).
        """
        async with self._lock:
            n = len(self.index)
            if n == 0:
                return []
            mat = np.stack(self.index.vectors)
            # normalize
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
            normed = mat / norms
            used = set()
            clusters = []
            for i in range(n):
                if i in used:
                    continue
                # start new cluster with i
                sims = (normed @ normed[i].T).reshape(-1)
                # pick top similar indices (including itself)
                order = np.argsort(-sims)
                sel = []
                for j in order:
                    if j in used:
                        continue
                    sel.append(j)
                    if len(sel) >= cluster_size:
                        break
                for j in sel:
                    used.add(j)
                clusters.append(sel)

            # build summaries via LLM
            summaries = []
            for sel in clusters:
                texts = [self.index.metadata[j].get("text", "") for j in sel]
                combined = "\n\n".join([t[:summary_max_chars] for t in texts])
                # call summarizer via LLM
                prompt = (
                    "Summarize the following related pieces of knowledge into a concise paragraph (1-3 sentences):\n\n"
                    f"{combined}\n\n"
                )
                try:
                    raw = await self.llm.generate(prompt)
                    summary = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
                except Exception:
                    summary = "[summary_failed]"
                # create cluster metadata
                cluster_meta = {
                    "summary": summary,
                    "members": [self.index.metadata[j] for j in sel],
                    "size": len(sel),
                    "ts": time.time()
                }
                summaries.append(cluster_meta)
            return summaries


# ------------------------------------------------------------
# MEMORY SYSTEM: integrates episodic + semantic + summarizer
# - add_episode(action, result)
# - add_knowledge(text, metadata)
# - retrieve_context(query, k_semantic, k_episodic)
# ------------------------------------------------------------
class MemorySystem:
    def __init__(
        self,
        llm_adapter,
        episodic_limit: int = 200,
        embed_dim: int = 768,
        sem_max_items: int = 5000,
        persistent: bool = False,
        index_path: Optional[str] = None
    ):
        
        # advanced episodic memory (auto summarization)
        self.episodic = EpisodicMemoryAdvanced(llm_adapter, max_items=episodic_limit, chunk_size=20, summary_retention=20)
        # advanced semantic memory (importance/forgetting)
        self.semantic = SemanticMemoryAdvanced(llm_adapter, dim=embed_dim, max_items=sem_max_items, persistent=persistent, index_path=index_path)
        self.llm = llm_adapter

    # Agent writes its steps (episodic)
    async def add_episode(self, action: str, result: Any):
        # store a stable serializable representation
        try:
            record = {"action": action, "result": result, "ts": time.time()}
        except Exception:
            # fallback to stringifying
            record = {"action": action, "result": str(result), "ts": time.time()}
        await self.episodic.add(record)

    # Agent stores long-term knowledge (semantic)
    async def add_knowledge(self, text: str, metadata: Dict[str, Any] = None):
        # sanitize/ensure text
        meta = dict(metadata or {})
        meta.setdefault("source", meta.get("source", "unknown"))
        meta.setdefault("task", meta.get("task", None))
        await self.semantic.add(text, meta)

    async def retrieve_context(self, query: str, k_semantic: int = 3, k_episodic: int = 3) -> Dict[str, Any]:
        """
        Returns a dict:
          {
             "semantic": [ {score, text, importance, ...}, ... ],
             "episodic": [ recent events... ],
             "episodic_summaries": [ summary blobs... ]
          }
        """
        # run semantic and episodic retrieval concurrently
        sem_task = asyncio.create_task(self.semantic.search(query, k=k_semantic))
        epi_task = asyncio.create_task(self._gather_episodic_context(k_episodic= k_episodic))
        sem = await sem_task
        epi_block = await epi_task
        return {
            "semantic": sem,
            "episodic": epi_block.get("recent", []),
            "episodic_summaries": epi_block.get("summaries", [])
        }

    async def _gather_episodic_context(self, k_episodic: int = 3) -> Dict[str, Any]:
        recent = self.episodic.search_last(k=k_episodic)
        summaries = self.episodic.get_recent_summaries(n=3)
        return {"recent": recent, "summaries": summaries}

    # helper: compact semantic memory by clustering & summarizing
    async def compact_semantic(self, cluster_size: int = 10):
        """
        Run a cluster+summarize pass; returned cluster summaries can be stored in semantic memory
        and original members pruned if desired.
        """
        cluster_summaries = await self.semantic.cluster_and_summarize(cluster_size=cluster_size)
        # store cluster summaries back into semantic memory as higher-importance nodes
        for cs in cluster_summaries:
            summary_text = cs.get("summary", "")
            meta = {"source": "semantic_cluster_summary", "size": cs.get("size", 0), "ts": cs.get("ts", time.time())}
            await self.semantic.add(summary_text, meta)

    # optional: snapshots / export
    async def export_state(self) -> Dict[str, Any]:
        return {
            "episodic_count": len(self.episodic.get_all()),
            "episodic_summaries": self.episodic.get_recent_summaries(n=10),
            "semantic_count": len(self.semantic.index),
        }

