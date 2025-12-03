# agentforge/core/llm.py
from __future__ import annotations
import os
import json
import time
import numpy as np
import asyncio
from typing import Any, Dict, Optional, AsyncIterator, List
from dataclasses import dataclass

# optional imports
try:
    import openai
except Exception:
    openai = None

try:
    import replicate
except Exception:
    replicate = None

try:
    import anthropic
except Exception:
    anthropic = None

try:
    import httpx
except Exception:
    httpx = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except Exception:
    # fallback
    def retry(*args, **kwargs):
        def _wrap(f): return f
        return _wrap
    stop_after_attempt = None
    wait_exponential = None
    retry_if_exception_type = None

# token counting
try:
    import tiktoken
except Exception:
    tiktoken = None


# --------------------------------------------------------
# LLMResponse
# --------------------------------------------------------
@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None


# --------------------------------------------------------
# BaseLLM interface
# --------------------------------------------------------
class BaseLLM:
    provider: str = "base"

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        raise NotImplementedError()

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        raise NotImplementedError()

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        raise NotImplementedError()

    # optional function-calling aware method
    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Adapter may implement this to return structured model outputs (e.g. OpenAI ChatCompletion dict).
        By default raise NotImplementedError to indicate unsupported.
        """
        raise NotImplementedError()


# --------------------------------------------------------
# LLM Stats
# --------------------------------------------------------
@dataclass
class LLMStats:
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


# --------------------------------------------------------
# LLMClient: caching + retry + stats
# --------------------------------------------------------
class LLMClient:
    def __init__(self, model: str, adapter_kwargs: Optional[dict] = None, cache: Optional[dict] = None):
        self.model = model
        self.adapter_kwargs = adapter_kwargs or {}
        self.adapter = self._make_adapter(model, **self.adapter_kwargs)
        self.stats = LLMStats()
        self._cache = cache or {}

    def _make_adapter(self, model: str, **kwargs) -> BaseLLM:
        prefix = model.split(":", 1)[0] if ":" in model else "openai"

        if prefix == "openai":
            return OpenAIAdapter(model, **kwargs)
        if prefix == "anthropic":
            return AnthropicAdapter(model, **kwargs)
        if prefix == "replicate":
            return ReplicateAdapter(model, **kwargs)
        if prefix == "huggingface":
            return HuggingFaceAdapter(model, **kwargs)
        if prefix in ("ollama", "local"):
            return OllamaAdapter(model, **kwargs)
        if prefix in ("mock", "test"):
            return MockAdapter(model)

        return MockAdapter(model)

    # ------------------------
    # Generate
    # ------------------------
    async def generate(self, prompt: str, use_cache: bool = True, cache_ttl: int = 300, **kwargs) -> LLMResponse:
        key = f"{self.model}|{prompt}|{json.dumps(kwargs, sort_keys=True)}"

        # cache hit
        if use_cache and key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["ts"] < cache_ttl:
                return entry["resp"]

        @retry(
            stop=stop_after_attempt(3) if stop_after_attempt else None,
            wait=wait_exponential(min=1, max=5) if wait_exponential else None,
            retry=retry_if_exception_type(Exception) if retry_if_exception_type else None,
        )
        async def _call():
            resp = await self.adapter.generate(prompt, **kwargs)
            self.stats.calls += 1
            if resp.tokens_used:
                self.stats.tokens += resp.tokens_used
            if resp.cost_usd:
                self.stats.cost_usd += resp.cost_usd
            return resp

        try:
            resp = await _call()
        except Exception:
            resp = await self.adapter.generate(prompt, **kwargs)

        if use_cache:
            self._cache[key] = {"ts": time.time(), "resp": resp}

        return resp

    # ------------------------
    # Function-calling aware generate (returns raw structured response)
    # ------------------------
    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], use_cache: bool = True, cache_ttl: int = 300, **kwargs) -> Any:
        """
        Preferred path for function-calling: returns the adapter's raw structured response
        (e.g. OpenAI ChatCompletion response dict). If adapter doesn't implement it, raise.
        """
        key = f"{self.model}|FUNC|{prompt}|{json.dumps(functions, sort_keys=True)}|{json.dumps(kwargs, sort_keys=True)}"
        if use_cache and key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["ts"] < cache_ttl:
                return entry["resp"]

        @retry(
            stop=stop_after_attempt(3) if stop_after_attempt else None,
            wait=wait_exponential(min=1, max=5) if wait_exponential else None,
            retry=retry_if_exception_type(Exception) if retry_if_exception_type else None,
        )
        async def _call():
            # delegate to adapter
            resp = await self.adapter.generate_with_functions(prompt, functions=functions, **kwargs)
            # don't try to update tokens/cost here (adapter may include it)
            return resp

        try:
            resp = await _call()
        except NotImplementedError:
            # adapter doesn't support it
            raise
        except Exception:
            # final fallback: raise to caller
            resp = await self.adapter.generate_with_functions(prompt, functions=functions, **kwargs)

        if use_cache:
            self._cache[key] = {"ts": time.time(), "resp": resp}
        return resp

    # ------------------------
    # Stream
    # ------------------------
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        async for chunk in self.adapter.stream(prompt, **kwargs):
            yield chunk

    # ------------------------
    # Embedding
    # ------------------------
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        return await self.adapter.embed(texts, **kwargs)



# --------------------------------------------------------
# HuggingFace Adapter (AI Gateway Compatible)
# --------------------------------------------------------
class HuggingFaceAdapter(BaseLLM):
    provider = "huggingface"

    def __init__(
        self,
        model: str = "huggingface:openai/gpt-oss-20b:groq",
        temperature: float = 0.0,
        api_key: Optional[str] = None
    ):
        if httpx is None:
            raise RuntimeError("httpx required for HuggingFace")

        self.api_key = api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.api_key:
            raise ValueError(
                "Hugging Face API key is required. "
                "Set HUGGINGFACEHUB_API_TOKEN or pass api_key."
            )

        # "huggingface:<model>" → "model"
        self.model_id = model.split(":", 1)[1] if ":" in model else model

        # HuggingFace AI Gateway endpoint (do not append model!)
        self.base_url = "https://router.huggingface.co/v1/chat/completions"

        self.temperature = temperature

    # --------------------------------------------------------
    # Non-streaming generation
    # --------------------------------------------------------
    async def generate(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> LLMResponse:
        async with httpx.AsyncClient(timeout=120) as client:
            headers = {"Authorization": f"Bearer {self.api_key}"}

            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
            }

            if max_tokens is not None:
                payload["max_tokens"] = max_tokens

            r = await client.post(self.base_url, json=payload, headers=headers)

            # Error check
            if r.status_code != 200:
                try:
                    error = r.json().get("error", r.text)
                except:
                    error = r.text
                raise RuntimeError(f"Hugging Face API error ({r.status_code}): {error}")

            data = r.json()

            # HF returns:
            # {
            #  "choices": [
            #      {"message": {"content": "..."}}
            #  ]
            # }
            try:
                text = data["choices"][0]["message"]["content"]
            except Exception:
                raise RuntimeError(f"Unexpected HuggingFace response format: {data}")

            return LLMResponse(text=text, raw=data)

    # --------------------------------------------------------
    # Pseudo-streaming (HF does not support streaming yet)
    # --------------------------------------------------------
    async def stream(self, prompt: str, max_tokens: Optional[int] = None, **kwargs):
        """
        Async streaming için HF AI Gateway SSE endpoint kullanımı.
        """

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model_id,
            "temperature": self.temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", self.base_url, headers={"Authorization": f"Bearer {self.api_key}"}, json=payload) as r:
                async for line_bytes in r.aiter_lines():
                    if not line_bytes.startswith("data:"):
                        continue
                    if line_bytes.strip() == "data: [DONE]":
                        return
                    try:
                        chunk = json.loads(line_bytes.lstrip("data:").strip())
                        # HF streaming delta token
                        delta = chunk["choices"][0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            yield text
                    except Exception:
                        continue
                    await asyncio.sleep(0)  # event loop rahat çalışsın
    
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Embed texts using Hugging Face's embedding-gemma-300m model via AI Gateway.
        Supports batch input.
        """
        if not isinstance(texts, list):
            texts = [texts]

        async with httpx.AsyncClient(timeout=120) as client:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": texts,
                "options": {"wait_for_model": True}
            }
            r = await client.post(
                "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5/pipeline/feature-extraction",
                json=payload,
                headers=headers
            )

            if r.status_code != 200:
                try:
                    error = r.json().get("error", r.text)
                except:
                    error = r.text
                raise RuntimeError(f"Hugging Face embedding API error ({r.status_code}): {error}")

            data = r.json()
            # HF embedding yanıtı formatı:
            # { "data": [ {"embedding": [...]}, ... ] }
            embeddings = []
            try:
                for item in data:
                    if isinstance(item, list) and len(item) == 1:
                        # Some HF pipelines nest extra dimension
                        embeddings.append(item[0])
                    else:
                        embeddings.append(item)

                # Normalize embeddings (recommended for BGE)
                def l2norm(v):
                    n = sum(x*x for x in v) ** 0.5
                    return [x / n for x in v]
                
                embeddings = [l2norm(e) for e in embeddings]

                return embeddings
            except Exception as e:
                raise RuntimeError(f"Unexpected Hugging Face embedding response format: {data}")

# --------------------------------------------------------
# Anthropic Adapter
# --------------------------------------------------------
class AnthropicAdapter(BaseLLM):
    provider = "anthropic"

    def __init__(self, model: str = "anthropic:claude-3-5-sonnet-latest", temperature: float = 0.0, api_key: Optional[str] = None):
        if anthropic is None:
            raise RuntimeError("anthropic package is not installed")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key."
            )
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.model = model.split(":", 1)[1] if ":" in model else model
        self.temperature = temperature

    async def generate(self, prompt: str, max_tokens: int = 4096, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = await self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            text = resp.content[0].text
            tokens = resp.usage.input_tokens + resp.usage.output_tokens
            return LLMResponse(text=text, raw=resp, tokens_used=tokens)
        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    async def stream(self, prompt: str, max_tokens: int = 4096, **kwargs) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        try:
            async with self.client.messages.stream(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            ) as stream:
                async for chunk in stream.text_stream:
                    yield chunk
        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic streaming API error: {e}")

    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], max_tokens: int = 4096, **kwargs) -> Any:
      """
      Anthropic tool use için generate_with_functions implementasyonu.
      OpenAI formatına uyumlu çıktı döner.
      """
      if anthropic is None:
          raise RuntimeError("anthropic package is not installed")
    
      # OpenAI formatındaki fonksiyonları Anthropic tool formatına dönüştür
      tools = []
      for func in functions:
         tool_def = {
           "name": func["name"],
           "description": func.get("description", ""),
           "input_schema": func.get("parameters", {})
         }
         tools.append(tool_def)
    
      messages = [{"role": "user", "content": prompt}]
    
      try:
          resp = await self.client.messages.create(
              model=self.model,
              messages=messages,
              max_tokens=max_tokens,
              temperature=self.temperature,
              tools=tools,
              tool_choice={"type": "auto"}  # Modelin tool seçmesine izin ver
          )
        
          # Anthropic yanıtını OpenAI formatına dönüştür
          # OpenAI formatı bekleniyor: {"choices": [{"message": {"function_call": {...}}}]
        
          # Tool çağrısı var mı kontrol et
          tool_calls = [block for block in resp.content if block.type == "tool_use"]
        
          if tool_calls:
              # İlk tool çağrısını al (çoğu durumda yeterli)
              tool_call = tool_calls[0]
              function_call = {
                  "name": tool_call.name,
                  "arguments": json.dumps(tool_call.input)  # input zaten dict formatında
              }
            
              # OpenAI formatına dönüştür
              openai_format = {
                  "choices": [
                      {
                          "message": {
                              "role": "assistant",
                              "content": None,
                              "function_call": function_call
                          },
                          "finish_reason": "function_call"
                      }
                  ],
                  "raw_anthropic_response": resp  # orijinal yanıtı debug için sakla
              } 
              return openai_format
          else:
              # Tool çağrısı yoksa, normal metin yanıtı
              text_content = [block.text for block in resp.content if block.type == "text"]
              full_text = "\n".join(text_content)
            
              return {
                  "choices": [
                      {
                          "message": {
                              "role": "assistant",
                              "content": full_text,
                              "function_call": None
                          },
                          "finish_reason": "stop"
                      }
                  ]
              }
      except anthropic.APIError as e:
          raise RuntimeError(f"Anthropic function calling failed: {e}")      
      except Exception as e:
          # Hata durumunda fallback
          print(f"Anthropic function calling failed: {e}")
          raise

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        raise NotImplementedError("Anthropic embeddings not implemented")

# --------------------------------------------------------
# Replicate Adapter
# --------------------------------------------------------
class ReplicateAdapter(BaseLLM):
    provider = "replicate"

    def __init__(self, model: str = "replicate:meta/llama3-70b-instruct", temperature: float = 0.0, api_key: Optional[str] = None):
        if replicate is None:
            raise RuntimeError("replicate package not installed")
        self.api_key = api_key or os.getenv("REPLICATE_API_TOKEN")
        if not self.api_key:
            raise ValueError(
                "Replicate API token is required. "
                "Set REPLICATE_API_TOKEN environment variable or pass api_key."
            )
        self.client = replicate.Client(api_token=self.api_key)
        self.model_id = model.split(":", 1)[1] if ":" in model else model
        self.temperature = temperature

    async def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> LLMResponse:
        loop = asyncio.get_event_loop()
        try:
            input_data = {
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            }
            output = await loop.run_in_executor(None, lambda: self.client.run(
                self.model_id,
                input=input_data
            ))
            text = "".join(output) if isinstance(output, list) else str(output)
            return LLMResponse(text=text, raw=output)
        except replicate.exceptions.ReplicateError as e:
            raise RuntimeError(f"Replicate API error: {e}")
        except Exception as e:
            raise RuntimeError(f"Replicate API error: {e}")
        
    async def stream(self, prompt: str, max_tokens: int = 512, **kwargs):
        """
        Gerçek Replicate streaming.
        replicate.stream() kullanarak token token yield eder.
        """
        try:
            input_data = {
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            }

            # Async stream başlat
            stream = self.client.stream(
                self.model_id,
                input=input_data
            )

            # Replicate event akışını async iterate ediyoruz
            async for event in stream:
                # Sadece token değişimlerini yakala
                if event.type == "text-delta":
                    delta = event.data  # Gelen token chunk
                    if delta:
                        yield delta

        except Exception as e:
            raise RuntimeError(f"Replicate streaming error: {e}")   
    
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Embed texts using a Replicate-hosted embedding model.
        By default, uses: "nateraw/bge-large-en-v1.5"
        """
        if not isinstance(texts, list):
            texts = [texts]

        # Default embedding model (lightweight, multilingual)
        embed_model = "nateraw/bge-large-en-v1.5:9cf9f015a9cb9c61d1a2610659cdac4a4ca222f2d3707a68517b18c198a9add1"

        loop = asyncio.get_event_loop()
        try:
            # Replicate **supports batching**: send entire list
            output = await loop.run_in_executor(
                None,
                lambda: self.client.run(
                    embed_model,
                    input={"texts": texts}
                )
            )
        except Exception as e:
            raise RuntimeError(f"Replicate embedding API failed: {e}")

        embeddings = []

        # Expected formats:
        #  1. [{"embedding": [...]}, ...]
        #  2. [[...], [...]]
        #  3. [[[...]], [[...]]] → extra nesting
        try:
            for item in output:
                # Case 1: {"embedding": [...]}
                if isinstance(item, dict) and "embedding" in item:
                    emb = item["embedding"]

                # Case 2: [[...]] → remove extra dimension
                elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], list):
                    emb = item[0]

                # Case 3: direct list vector
                elif isinstance(item, list):
                    emb = item

                else:
                    raise ValueError(f"Unexpected embedding output format: {item}")

                embeddings.append(emb)

            # Optional but recommended for BGE embeddings → **L2 normalize**
            def l2norm(v):
                n = sum(x * x for x in v) ** 0.5
                return [x / n for x in v]

            embeddings = [l2norm(e) for e in embeddings]

            return embeddings

        except Exception as e:
            raise RuntimeError(f"Error parsing Replicate embedding output: {e}")
    
# --------------------------------------------------------
# OpenAI Adapter
# --------------------------------------------------------
class OpenAIAdapter(BaseLLM):
    provider = "openai"

    def __init__(
        self,
        model: str = "openai:gpt-3.5-turbo",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        if openai is None:
            raise RuntimeError("openai package is not installed")
        
        # API key'i öncelikle argümandan al, sonra ortam değişkeninden
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key."
            )
        
        # Modern async istemciyi oluştur (global openai.api_key KULLANMA!)
        self.client = openai.AsyncOpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model.split(":", 1)[1] if ":" in model else model
        self.temperature = temperature

    async def generate(self, prompt: str, max_tokens: int = 512, functions: Optional[List[Dict[str, Any]]] = None, **kwargs) -> LLMResponse:
        if functions:
            raw = await self.generate_with_functions(prompt, functions=functions, max_tokens=max_tokens, **kwargs)
            try:
                choices = raw.get("choices", []) or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if content:
                        text = content
                    else:
                        text = json.dumps(raw)
                else:
                    text = json.dumps(raw)
            except Exception:
                text = json.dumps(raw)
            return LLMResponse(text=text, raw=raw)

        # Modern async çağrı
        messages = [{"role": "user", "content": prompt}]
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else None
        return LLMResponse(text=text, raw=resp, tokens_used=tokens)

    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], max_tokens: int = 512, **kwargs) -> Any:
        messages = [{"role": "user", "content": prompt}]
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
            tools=[{"type": "function", "function": f} for f in functions],
            tool_choice="auto",
        )
        # OpenAI v1+ zaten dict benzeri nesne döner, planner bunu doğrudan işleyebilir
        return resp.model_dump()

    async def stream(self, prompt: str, max_tokens: int = 512, **kwargs) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(self, texts: List[str], model: str = "text-embedding-3-small", **kwargs) -> List[List[float]]:
        resp = await self.client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in resp.data]


# --------------------------------------------------------
# Ollama Adapter
# --------------------------------------------------------
class OllamaAdapter(BaseLLM):
    provider = "ollama"

    def __init__(self, model: str = "ollama:qwen3:4b", base_url: str = "http://localhost:11434", temperature: float = 0.0):
        if httpx is None:
            raise RuntimeError("httpx package required for Ollama")

        self.model = model.split(":", 1)[1] if ":" in model else model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    async def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> LLMResponse:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                },
            )
            r.raise_for_status()
            data = r.json()
            text = data.get("output") or data.get("text") or ""
            return LLMResponse(text=text, raw=data)

    async def stream(self, prompt: str, max_tokens: int = 512, **kwargs) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model, 
                    "prompt": prompt, 
                    "max_tokens": max_tokens, 
                    "temperature": self.temperature,
                    "stream": True
                },
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk.decode("utf-8", errors="ignore")

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Embed texts using a default local embedding model from Ollama.
        Automatically uses 'nomic-embed-text' (recommended, free, high-quality).
        No need for user to specify embedding model.
        """
        if not isinstance(texts, list):
            texts = [texts]

        # Sabit, öntanımlı embedding modeli — kullanıcıdan gizli
        EMBEDDING_MODEL = "nomic-embed-text"  

        async with httpx.AsyncClient(timeout=120) as client:
            embeddings = []
            for text in texts:
                try:
                    r = await client.post(
                        f"{self.base_url}/api/embed",
                        json={
                            "model": EMBEDDING_MODEL,
                            "input": text
                        }
                    )
                    r.raise_for_status()
                    data = r.json()
                    if "embeddings" in data and len(data["embeddings"]) > 0:
                        emb = data["embeddings"][0]
                        embeddings.append(emb)
                    else:
                        raise RuntimeError(f"Ollama embedding returned no data: {data}")
                except Exception as e:
                    # Fallback: empty embedding (veya hata fırlat)
                    # Alternatif: embedding yapmadan geç (semantic memory devre dışı)
                    raise RuntimeError(f"Ollama embedding failed. Is '{EMBEDDING_MODEL}' pulled? Run: `ollama pull nomic-embed-text`")
            return embeddings


# --------------------------------------------------------
# Mock Adapter (for tests)
# --------------------------------------------------------
class MockAdapter(BaseLLM):
    provider = "mock"

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(text=f"[MOCK] {prompt}", raw={"mock": True})

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        for i in range(3):
            await asyncio.sleep(0.02)
            yield f"[MOCK_CHUNK_{i}] "
        yield "[END]"

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        return [[float(len(t))] * 8 for t in texts]  # simple deterministic embedding

    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Minimal mock behavior: return a fake structured response that *chooses* to call the first function
        with an example argument. This helps unit tests.
        """
        if not functions:
            return {"choices": [{"message": {"content": "[MOCK] no functions"}}]}
        func = functions[0]
        # craft a fake function_call structure like OpenAI returns
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": func.get("name"),
                            "arguments": json.dumps({list(func.get("parameters", {}).get("properties", {}).keys())[0]: "example"})
                        }
                    },
                    "finish_reason": "function_call"
                }
            ]
        }


# --------------------------------------------------------
# LLMAdapter (simple wrapper for LLMClient)
# --------------------------------------------------------
class LLMAdapter:
    """
    This is the unified interface used by the Agent:
    - generate() -> returns text
    - stream() -> async generator of tokens
    - embed() -> vector embeddings
    - generate_with_functions() -> returns structured raw response (adapter-dependent)
    """

    def __init__(self, model: str, max_tokens: Optional[int] = None, **kwargs):
        self.client = LLMClient(model, adapter_kwargs=kwargs)
        self.model = model
        self.max_tokens = max_tokens

    async def generate(self, prompt: str, **kwargs) -> str:
        if "max_tokens" not in kwargs and self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        resp = await self.client.generate(prompt, **kwargs)
        # client.generate returns an LLMResponse
        return resp.text if isinstance(resp, LLMResponse) else str(resp)

    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Exposes the client's generate_with_functions (returns structured raw response from adapter)
        """
        return await self.client.generate_with_functions(prompt, functions=functions, **kwargs)

    async def stream(self, prompt: str, **kwargs):
        if "max_tokens" not in kwargs and self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        async for token in self.client.stream(prompt, **kwargs):
            yield token

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        return await self.client.embed(texts, **kwargs)


# ---------------------------
# Multi-LLM Orchestration / Ensemble
# ---------------------------
class MultiLLMManager:
    """
    Orchestrates multiple LLMAdapters:
      - ensemble: majority vote / first responder / weighted
      - failover: primary -> fallback models
      - tool-prompt adaptation: functions routed per LLM
    """
    def __init__(self, llms: List[LLMAdapter]):
        if not llms:
            raise ValueError("At least one LLMAdapter required")
        self.llms: List[LLMAdapter] = llms

    async def generate(self, prompt: str, strategy: str = "first", **kwargs) -> str:
        """
        strategy:
          - 'first' : first model that succeeds
          - 'ensemble' : aggregate responses from all models (concatenate)
        """
        if strategy == "first":
            for llm in self.llms:
                try:
                    return await llm.generate(prompt, **kwargs)
                except Exception:
                    continue
            raise RuntimeError("All LLMs failed")
        elif strategy == "ensemble":
            results = []
            for llm in self.llms:
                try:
                    text = await llm.generate(prompt, **kwargs)
                    results.append(text)
                except Exception:
                    continue
            return "\n".join(results)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], strategy: str = "first", **kwargs) -> Any:
        if strategy == "first":
            for llm in self.llms:
                try:
                    resp = await llm.generate_with_functions(prompt, functions, **kwargs)
                    return resp
                except NotImplementedError:
                    continue
                except Exception:
                    continue
            raise RuntimeError("All LLMs failed or no function-calling support")
        elif strategy == "ensemble":
            # collect multiple structured responses
            responses = []
            for llm in self.llms:
                try:
                    resp = await llm.generate_with_functions(prompt, functions, **kwargs)
                    responses.append(resp)
                except Exception:
                    continue
            return responses
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    async def stream(self, prompt: str, strategy: str = "first", **kwargs) -> AsyncIterator[str]:
        if strategy == "first":
            for llm in self.llms:
                try:
                    async for chunk in llm.stream(prompt, **kwargs):
                        yield chunk
                    return
                except Exception:
                    continue
            raise RuntimeError("All LLM streams failed")
        elif strategy == "ensemble":
            queues = [asyncio.Queue() for _ in self.llms]
            async def stream_llm(i, llm, q):
                try:
                    async for c in llm.stream(prompt, **kwargs):
                        await q.put(c)
                finally:
                    await q.put(None)
            tasks = [asyncio.create_task(stream_llm(i, llm, q)) for i, (llm, q) in enumerate(zip(self.llms, queues))]
            finished = [False] * len(self.llms)
            while not all(finished):
                for i, q in enumerate(queues):
                    if finished[i]:
                        continue
                    try:
                        c = q.get_nowait()
                        if c is None:
                            finished[i] = True
                        else:
                            yield c
                    except asyncio.QueueEmpty:
                        continue
                await asyncio.sleep(0.01)
            await asyncio.gather(*tasks)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    async def embed(self, texts: List[str], strategy: str = "first", **kwargs) -> List[List[float]]:
        if strategy == "first":
            return await self.llms[0].embed(texts, **kwargs)
        elif strategy == "ensemble":
            # average embeddings
            embs = []
            for llm in self.llms:
                try:
                    e = await llm.embed(texts, **kwargs)
                    embs.append(e)
                except Exception:
                    continue
            if not embs:
                raise RuntimeError("All embedding calls failed")
            embs_np = [np.array(e) for e in embs]
            avg = np.mean(np.stack(embs_np), axis=0)
            return avg.tolist()
        else:
            raise ValueError(f"Unknown strategy {strategy}")


# --------------------------------------------------------
# Token counting helper
# --------------------------------------------------------
def count_tokens(model: str, text: str) -> Optional[int]:
    if tiktoken is None:
        return None
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
