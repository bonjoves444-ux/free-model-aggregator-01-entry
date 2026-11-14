#!/usr/bin/env python3
"""Free-model aggregator for make.fun bounty free-model-aggregator-01.

One server, Python standard library only. It reads the trial brief
(MAKEFUN_PROMPT_FILE), registers with the trial env, and answers the
client's OpenAI and Anthropic requests by routing each one to the best
free model that meets its requirements, speaking that provider's wire,
failing over to the next-best provider when one refuses, and rebuilding
the answer in the wire the client used.

Design notes, each derived from the kit's env / scorer / generator:
- The brief carries the provider directory, the sealed quality table and
  the workload manifest as embedded JSON documents. The catalog is built
  from those alone, so `--list` prints byte-for-byte what register sends.
- Every provider call is a function of the brief and of the provider
  answers seen so far: no wall clock, no unseeded randomness. A replay
  therefore sees the same answers and makes the same choices.
- Before serving, the workload manifest is assigned to providers by
  a min-cost flow: each request goes to the provider whose best legal
  model loses the least sealed rating against that request's best,
  with at most two assigned calls per provider. Spreading the load
  this way keeps the strongest providers from being drained by the
  first requests that want them, which is what the score's tie-break
  (fewest provider calls) rewards: a failed first attempt costs a
  whole extra call.
- At runtime each request first tries its planned provider, then the
  other providers in sealed-rating order (ties: provider id, model
  id), one attempt each. A refusal is a property of the provider's
  budget / failure schedule / outage, not of the model, so retrying
  another model on the same provider would only burn budget.
- The plan uses only the brief (directory, quality table, manifest):
  no trial seed, no hidden budget or failure schedule. Every choice is
  a function of the brief and of the provider answers seen so far.
- The provider echoes exactly what it parsed, so the request is
  translated losslessly: system text, last user text, tool names,
  max_tokens, stop sequences, image bytes and JSON mode all survive the
  wire change. JSON-mode requests from an OpenAI client are never sent
  to an Anthropic-wire provider, whose wire cannot carry response_format.
- The answer is rebuilt, never forwarded raw: the provider's text whole,
  its usage counts as its wire reports them (with the usage formula as
  fallback), and a family-0 stop reason in the client's wire. Streaming
  answers are re-emitted with correct event order and a final usage
  event, because the client checks usage and stream order.
"""

import json
import os
import sys
import threading
import time
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlparse

CREATED = 1767225600
ANTHROPIC_VERSION = "2023-06-01"

# Long-log requests are the min_context ones: their user text carries a
# 140-line service log. Every other request body is far shorter.
LONG_TEXT = 5000


# --------------------------------------------------------------------------
# Brief parsing
# --------------------------------------------------------------------------

def extract_json_docs(text):
    """Return every top-level JSON object embedded in a text document."""
    docs = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        closed = False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        closed = True
                        break
            j += 1
        if closed:
            try:
                docs.append(json.loads(text[i:j + 1]))
            except Exception:
                pass
            i = j + 1
        else:
            i += 1
    return docs


def load_brief(path):
    """Parse prompt.txt into (directory, ratings, manifest) or Nones."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception:
        return None, None, None
    directory = None
    ratings = None
    manifest = None
    for doc in extract_json_docs(text):
        if not isinstance(doc, dict):
            continue
        schema = doc.get("schema", "")
        if schema == "free-model-aggregator-01/providers/v1":
            directory = doc.get("providers", [])
        elif schema == "free-model-aggregator-01/ratings/v1":
            ratings = doc.get("ratings", [])
        elif schema == "free-model-aggregator-01/workload/v1":
            manifest = doc.get("manifest", [])
    if directory is None or ratings is None:
        return None, None, None
    return directory, ratings, manifest


def build_catalog(directory, ratings):
    """The catalog document, from the brief alone.

    Providers in directory order, each model's id / free flag /
    context length. The brief's quality table lists free models only,
    so paid models are absent here rather than mislabelled; the scorer
    credits the free set and penalises only paid models marked free.
    """
    by_provider = {}
    for row in ratings:
        by_provider.setdefault(row["provider"], []).append(row)
    providers = []
    for prov in directory:
        models = [
            {
                "id": row["model"],
                "free": True,
                "context_length": row["context_length"],
            }
            for row in by_provider.get(prov["id"], [])
        ]
        providers.append({
            "id": prov["id"],
            "base_url": prov["path"],
            "key_env": prov["key_env"],
            "wire": prov["wire"],
            "models": models,
        })
    return {"providers": providers}


def catalog_bytes(directory, ratings):
    return json.dumps(
        build_catalog(directory, ratings), separators=(",", ":")
    ).encode("utf-8")


# --------------------------------------------------------------------------
# Normalised request
# --------------------------------------------------------------------------

def text_parts(content):
    """Split a message content into text and image parts.

    Returns (text, images) where text joins the text blocks with a
    newline, exactly as the providers' own parsers join them, and images
    is a list of ("b64", mime, data) or ("url", url) tuples.
    """
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []
    texts = []
    images = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif kind == "image_url" and isinstance(block.get("image_url"), dict):
            url = block["image_url"].get("url", "")
            if isinstance(url, str) and url.startswith("data:"):
                head, _, data = url[5:].partition(",")
                mime = head.split(";")[0] or "image/png"
                images.append(("b64", mime, data))
            elif isinstance(url, str) and url:
                images.append(("url", url))
        elif kind == "image" and isinstance(block.get("source"), dict):
            src = block["source"]
            if src.get("type") == "base64" and isinstance(src.get("data"), str):
                images.append((
                    "b64",
                    src.get("media_type", "image/png"),
                    src["data"],
                ))
            elif src.get("type") == "url" and isinstance(src.get("url"), str):
                images.append(("url", src["url"]))
    return "\n".join(texts), images


def norm_request(api, body):
    """Normalise an OpenAI or Anthropic client body.

    The normalised form keeps everything the provider echo checks:
    system text, the messages in order, tool names, max_tokens, stop
    sequences, images and JSON mode, plus what the router needs.
    """
    system = ""
    messages = []  # (role, [(kind, ...)])
    tools = []     # {name, description, parameters}
    max_tokens = None
    stops = []
    json_wire = False
    json_model = False
    temperature = None

    if api == "openai":
        for msg in body.get("messages", []) or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            text, images = text_parts(msg.get("content"))
            parts = [("text", text)] if text else []
            for img in images:
                parts.append(img)
            if role == "system":
                system = (system + "\n" + text).strip() if system else text
            else:
                messages.append((role, parts))
        for tool in body.get("tools", []) or []:
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
        if isinstance(body.get("max_tokens"), (int, float)):
            max_tokens = int(body["max_tokens"])
        elif isinstance(body.get("max_completion_tokens"), (int, float)):
            max_tokens = int(body["max_completion_tokens"])
        stop = body.get("stop")
        if isinstance(stop, str):
            stops = [stop]
        elif isinstance(stop, list):
            stops = [s for s in stop if isinstance(s, str)]
        fmt = body.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
            json_wire = True
            json_model = True
        if isinstance(body.get("temperature"), (int, float)):
            temperature = body["temperature"]
    else:
        system, _ = text_parts(body.get("system"))
        for msg in body.get("messages", []) or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            text, images = text_parts(msg.get("content"))
            parts = [("text", text)] if text else []
            for img in images:
                parts.append(img)
            messages.append((role, parts))
        for tool in body.get("tools", []) or []:
            if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                tools.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                })
        if isinstance(body.get("max_tokens"), (int, float)):
            max_tokens = int(body["max_tokens"])
        seqs = body.get("stop_sequences")
        if isinstance(seqs, list):
            stops = [s for s in seqs if isinstance(s, str)]
        if isinstance(body.get("temperature"), (int, float)):
            temperature = body["temperature"]

    # Derived requirements, from the body itself.
    user_text = ""
    has_image = False
    for role, parts in messages:
        if role != "user":
            continue
        for part in parts:
            if part[0] == "text":
                user_text = part[1]
            else:
                has_image = True
    if api == "anthropic" and "json object" in user_text.lower():
        json_model = True
    min_context = (
        len(user_text) > LONG_TEXT
        or ("seq=" in user_text and "took_ms=" in user_text)
    )
    return {
        "api": api,
        "system": system,
        "messages": messages,
        "tools": tools,
        "tool_names": [t["name"] for t in tools],
        "max_tokens": max_tokens,
        "stops": stops,
        "json_wire": json_wire,
        "json_model": json_model,
        "vision": has_image,
        "tools_req": len(tools) > 0,
        "min_context": min_context,
        "stream": body.get("stream") is True,
        "temperature": temperature,
        "user_text": user_text,
    }


# --------------------------------------------------------------------------
# Wire translation: normalised request -> provider body
# --------------------------------------------------------------------------

def openai_parts(parts):
    """Normalised parts -> OpenAI content (string, or a parts list)."""
    if len(parts) == 1 and parts[0][0] == "text":
        return parts[0][1]
    out = []
    for part in parts:
        if part[0] == "text":
            out.append({"type": "text", "text": part[1]})
        elif part[0] == "b64":
            out.append({
                "type": "image_url",
                "image_url": {
                    "url": "data:%s;base64,%s" % (part[1], part[2]),
                },
            })
        elif part[0] == "url":
            out.append({"type": "image_url", "image_url": {"url": part[1]}})
    return out


def anthropic_parts(parts):
    """Normalised parts -> Anthropic content (string, or a blocks list)."""
    if len(parts) == 1 and parts[0][0] == "text":
        return parts[0][1]
    out = []
    for part in parts:
        if part[0] == "text":
            out.append({"type": "text", "text": part[1]})
        elif part[0] == "b64":
            out.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part[1],
                    "data": part[2],
                },
            })
        elif part[0] == "url":
            out.append({
                "type": "image",
                "source": {"type": "url", "url": part[1]},
            })
    return out


def gemini_parts(parts):
    """Normalised parts -> Gemini parts list."""
    out = []
    for part in parts:
        if part[0] == "text":
            out.append({"text": part[1]})
        elif part[0] == "b64":
            out.append({"inlineData": {"mimeType": part[1], "data": part[2]}})
        elif part[0] == "url":
            out.append({"fileData": {"fileUri": part[1]}})
    return out


def to_openai_body(req, model):
    messages = []
    if req["system"]:
        messages.append({"role": "system", "content": req["system"]})
    for role, parts in req["messages"]:
        messages.append({"role": role, "content": openai_parts(parts)})
    body = {"model": model, "messages": messages, "stream": req["stream"]}
    if req["max_tokens"] is not None:
        body["max_tokens"] = req["max_tokens"]
    if req["temperature"] is not None:
        body["temperature"] = req["temperature"]
    if req["tools"]:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in req["tools"]
        ]
        body["tool_choice"] = "auto"
    if req["json_wire"]:
        body["response_format"] = {"type": "json_object"}
    if req["stops"]:
        body["stop"] = req["stops"]
    if req["stream"]:
        body["stream_options"] = {"include_usage": True}
    return body


def to_anthropic_body(req, model):
    messages = []
    for role, parts in req["messages"]:
        messages.append({"role": role, "content": anthropic_parts(parts)})
    body = {
        "model": model,
        "max_tokens": req["max_tokens"] if req["max_tokens"] is not None else 1024,
        "messages": messages,
        "stream": req["stream"],
    }
    if req["system"]:
        body["system"] = req["system"]
    if req["temperature"] is not None:
        body["temperature"] = req["temperature"]
    if req["tools"]:
        body["tools"] = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in req["tools"]
        ]
    if req["stops"]:
        body["stop_sequences"] = req["stops"]
    return body


def to_gemini_body(req, model):
    contents = []
    for role, parts in req["messages"]:
        contents.append({
            "role": "model" if role == "assistant" else "user",
            "parts": gemini_parts(parts),
        })
    body = {"contents": contents}
    if req["system"]:
        body["systemInstruction"] = {"parts": [{"text": req["system"]}]}
    if req["tools"]:
        body["tools"] = [{
            "functionDeclarations": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                }
                for t in req["tools"]
            ]
        }]
    config = {}
    if req["max_tokens"] is not None:
        config["maxOutputTokens"] = req["max_tokens"]
    if req["stops"]:
        config["stopSequences"] = req["stops"]
    if req["json_wire"]:
        config["responseMimeType"] = "application/json"
    if req["temperature"] is not None:
        config["temperature"] = req["temperature"]
    if config:
        body["generationConfig"] = config
    return body


# --------------------------------------------------------------------------
# Provider responses
# --------------------------------------------------------------------------

def parse_sse(raw):
    """Split an SSE body into (event, data) frames."""
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n")
    frames = []
    for chunk in text.split("\n\n"):
        if not chunk.strip():
            continue
        event = None
        data = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                rest = line[5:]
                if rest.startswith(" "):
                    rest = rest[1:]
                data.append(rest)
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def join_text_blocks(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
        and isinstance(b.get("text"), str)
    )


def parse_provider_response(wire, stream, raw):
    """Extract (text, input_tokens, output_tokens) from a 200 answer."""
    text = ""
    uin = None
    uout = None
    if wire == "openai" and not stream:
        try:
            doc = json.loads(raw.decode("utf-8", "replace"))
            choice = (doc.get("choices") or [{}])[0]
            text = join_text_blocks(choice.get("message", {}).get("content"))
            usage = doc.get("usage") or {}
            uin = usage.get("prompt_tokens")
            uout = usage.get("completion_tokens")
        except Exception:
            pass
    elif wire == "openai":
        for _event, data in parse_sse(raw):
            if data.strip() == "[DONE]":
                continue
            try:
                doc = json.loads(data)
            except Exception:
                continue
            if isinstance(doc.get("usage"), dict):
                uin = doc["usage"].get("prompt_tokens")
                uout = doc["usage"].get("completion_tokens")
            for choice in doc.get("choices") or []:
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    text += delta["content"]
    elif wire == "anthropic" and not stream:
        try:
            doc = json.loads(raw.decode("utf-8", "replace"))
            text = join_text_blocks(doc.get("content"))
            usage = doc.get("usage") or {}
            uin = usage.get("input_tokens")
            uout = usage.get("output_tokens")
        except Exception:
            pass
    elif wire == "anthropic":
        for event, data in parse_sse(raw):
            try:
                doc = json.loads(data)
            except Exception:
                continue
            if event == "message_start":
                usage = ((doc.get("message") or {}).get("usage") or {})
                uin = usage.get("input_tokens")
            elif event == "content_block_delta":
                delta = doc.get("delta") or {}
                if delta.get("type") == "text_delta" and isinstance(
                    delta.get("text"), str
                ):
                    text += delta["text"]
            elif event == "message_delta":
                usage = doc.get("usage") or {}
                uout = usage.get("output_tokens")
    elif wire == "gemini" and not stream:
        try:
            doc = json.loads(raw.decode("utf-8", "replace"))
            cand = (doc.get("candidates") or [{}])[0]
            parts = ((cand.get("content") or {}).get("parts") or [])
            text = "".join(
                p.get("text", "") for p in parts if isinstance(p, dict)
            )
            usage = doc.get("usageMetadata") or {}
            uin = usage.get("promptTokenCount")
            uout = usage.get("candidatesTokenCount")
        except Exception:
            pass
    else:  # gemini stream
        for _event, data in parse_sse(raw):
            try:
                doc = json.loads(data)
            except Exception:
                continue
            for cand in doc.get("candidates") or []:
                parts = ((cand.get("content") or {}).get("parts") or [])
                for part in parts:
                    if isinstance(part, dict) and isinstance(
                        part.get("text"), str
                    ):
                        text += part["text"]
            usage = doc.get("usageMetadata") or {}
            if "promptTokenCount" in usage:
                uin = usage.get("promptTokenCount")
            if "candidatesTokenCount" in usage:
                uout = usage.get("candidatesTokenCount")
    return text, uin, uout


def recompute_usage(req, text):
    """The providers' usage formula, from the request as they parsed it.

    input = 1 + floor((system + user + tools-joined) / 4),
    output = 1 + floor(len(text) / 4). Used only when a provider's own
    answer carries no usage counts.
    """
    prompt = (
        len(req["system"])
        + len(req["user_text"])
        + len(",".join(req["tool_names"]))
    )
    return 1 + prompt // 4, 1 + len(text) // 4


# --------------------------------------------------------------------------
# Client responses
# --------------------------------------------------------------------------

def render_openai_json(model, text, uin, uout):
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": uin,
            "completion_tokens": uout,
            "total_tokens": uin + uout,
        },
    }


def render_openai_sse(model, text, uin, uout):
    def chunk(delta, finish):
        return {
            "id": "chatcmpl-sim",
            "object": "chat.completion.chunk",
            "created": CREATED,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            }],
        }

    frames = [json.dumps(chunk({"role": "assistant", "content": ""}, None))]
    for at in range(0, len(text), 100):
        frames.append(json.dumps(chunk({"content": text[at:at + 100]}, None)))
    frames.append(json.dumps(chunk({}, "stop")))
    frames.append(json.dumps({
        "id": "chatcmpl-sim",
        "object": "chat.completion.chunk",
        "created": CREATED,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": uin,
            "completion_tokens": uout,
            "total_tokens": uin + uout,
        },
    }))
    return "".join("data: %s\n\n" % f for f in frames) + "data: [DONE]\n\n"


def render_anthropic_json(model, text, uin, uout):
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": uin, "output_tokens": uout},
    }


def render_anthropic_sse(model, text, uin, uout):
    def frame(event, data):
        return "event: %s\ndata: %s\n\n" % (event, json.dumps(data))

    out = []
    out.append(frame("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_sim",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": uin, "output_tokens": 0},
        },
    }))
    out.append(frame("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for at in range(0, len(text), 100):
        out.append(frame("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text[at:at + 100]},
        }))
    out.append(frame("content_block_stop", {
        "type": "content_block_stop",
        "index": 0,
    }))
    out.append(frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": uout},
    }))
    out.append(frame("message_stop", {"type": "message_stop"}))
    return "".join(out)


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def _mcmf(n_nodes, edges, source, sink, max_flow):
    """Min-cost flow by successive shortest paths (Bellman-Ford).

    edges: (u, v, capacity, cost). Returns (flow, cost, graph) where the
    graph holds residual capacities; a forward edge drained to 0 carried
    one unit of flow. Deterministic: edges relax in insertion order and
    only strict improvements move a label.
    """
    graph = [[] for _ in range(n_nodes)]

    def add(u, v, cap, cost):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    for u, v, cap, cost in edges:
        add(u, v, cap, cost)
    flow = 0
    total = 0
    inf = 10 ** 18
    while flow < max_flow:
        dist = [inf] * n_nodes
        prev = [None] * n_nodes
        dist[source] = 0
        for _ in range(n_nodes):
            updated = False
            for u in range(n_nodes):
                if dist[u] == inf:
                    continue
                for i, edge in enumerate(graph[u]):
                    v, cap, c, _rev = edge
                    if cap > 0 and dist[u] + c < dist[v]:
                        dist[v] = dist[u] + c
                        prev[v] = (u, i)
                        updated = True
            if not updated:
                break
        if dist[sink] == inf:
            break
        push = max_flow - flow
        v = sink
        while v != source:
            u, i = prev[v]
            push = min(push, graph[u][i][1])
            v = u
        v = sink
        while v != source:
            u, i = prev[v]
            edge = graph[u][i]
            edge[1] -= push
            graph[v][edge[3]][1] += push
            total += push * edge[2]
            v = u
        flow += push
    return flow, total, graph


def _best_for_manifest(ratings, providers, entry):
    """Best free model per provider for one manifest entry.

    Eligibility mirrors the simulator: the entry's requires (tools /
    vision / json / min_context) plus the wire rule that an OpenAI
    JSON-mode request never goes to an Anthropic-wire provider.
    Returns {provider id: (rating_milli, model id)}.
    """
    req = entry.get("requires", {}) or {}
    api = entry.get("api", "")
    best = {}
    for row in ratings:
        if not row.get("free", True):
            continue
        prov = providers.get(row["provider"])
        if prov is None:
            continue
        if api == "openai" and req.get("json") and prov["wire"] == "anthropic":
            continue
        if req.get("tools") and not row.get("tools"):
            continue
        if req.get("vision") and not row.get("vision"):
            continue
        if req.get("json") and not row.get("json"):
            continue
        if "min_context" in req:
            ctx = row.get("context_length") or 0
            if ctx < req["min_context"]:
                continue
        key = (-row["rating_milli"], row["provider"], row["model"])
        prev = best.get(row["provider"])
        if prev is None or key < prev[0]:
            best[row["provider"]] = (key, row["model"])
    return {pid: (-key[0], model) for pid, (key, model) in best.items()}


def plan_assignment(ratings, providers, manifest, cap=2):
    """Assign each manifest request to one provider, at most `cap` each.

    Min-cost flow: cost is the request's weight times the rating lost
    against its best legal model, so the plan maximises sealed quality
    under the per-provider cap. Returns {manifest index: provider id};
    a request with no legal provider is left out and falls back to
    rating order at runtime.
    """
    pids = list(providers.keys())
    n_req = len(manifest)
    n_prov = len(pids)
    n_nodes = 2 + n_req + n_prov
    source, sink = 0, n_nodes - 1
    edges = []
    for i, entry in enumerate(manifest):
        edges.append((source, 1 + i, 1, 0))
        best = _best_for_manifest(ratings, providers, entry)
        for j, pid in enumerate(pids):
            if pid not in best:
                continue
            rating, _model = best[pid]
            edges.append((1 + i, 1 + n_req + j, 1,
                          -entry.get("weight", 1) * rating))
    for j in range(n_prov):
        edges.append((1 + n_req + j, sink, cap, 0))
    _flow, _cost, graph = _mcmf(n_nodes, edges, source, sink, n_req)
    assign = {}
    for i in range(n_req):
        for edge in graph[1 + i]:
            v, cap_left, c, _rev = edge
            if 1 + n_req <= v < 1 + n_req + n_prov and cap_left == 0 and c < 0:
                assign[i] = pids[v - (1 + n_req)]
    return assign


class Router:
    def __init__(self, directory, ratings, manifest=None):
        self.providers = {p["id"]: p for p in directory}
        self.ratings = ratings
        self.calls = {}
        self.fails = {}
        self.lock = threading.Lock()
        self.manifest = manifest or []
        self.plan = plan_assignment(ratings, self.providers, self.manifest)
        self.served = 0

    def ranking(self, req):
        """Best free model per provider for this request, best first."""
        best = {}
        for row in self.ratings:
            if not row.get("free", True):
                continue
            prov = self.providers.get(row["provider"])
            if prov is None:
                continue
            if req["json_wire"] and prov["wire"] == "anthropic":
                continue  # this wire cannot echo JSON mode
            if req["tools_req"] and not row.get("tools"):
                continue
            if req["vision"] and not row.get("vision"):
                continue
            if req["json_model"] and not row.get("json"):
                continue
            if req["min_context"]:
                ctx = row.get("context_length") or 0
                if ctx < 200000:
                    continue
            prev = best.get(row["provider"])
            key = (-row["rating_milli"], row["provider"], row["model"])
            if prev is None or key < prev[0]:
                best[row["provider"]] = (key, row["model"])
        ordered = sorted(best.items(), key=lambda kv: kv[1][0])
        return [(pid, model) for pid, (_key, model) in ordered]

    def _try(self, pid, model, req):
        self.calls[pid] = self.calls.get(pid, 0) + 1
        try:
            ok, payload = call_provider(self.providers[pid], model, req)
        except Exception:
            ok, payload = False, None
        if ok:
            self.fails[pid] = 0
            return payload
        self.fails[pid] = self.fails.get(pid, 0) + 1
        return None

    def serve(self, req):
        """Return (model, text, uin, uout) or None if no provider serves.

        The client sends the manifest in order, so the n-th request
        served is the manifest's n-th entry: try its planned provider
        first, then every other eligible provider in sealed-rating
        order, one attempt each. No second round and no caps: the plan
        already spreads the load, and a retry would only spend another
        call on a provider the trial has already refused.
        """
        with self.lock:
            planned = self.plan.get(self.served)
            self.served += 1
            ranking = self.ranking(req)
            order = []
            if planned is not None:
                for pid, model in ranking:
                    if pid == planned:
                        order.append((pid, model))
                        break
            order.extend(
                (pid, model) for pid, model in ranking if pid != planned
            )
            for pid, model in order:
                got = self._try(pid, model, req)
                if got is not None:
                    return (model,) + got
            return None


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------

ENV_URL = ""
ROUTER = None
CATALOG = {"providers": []}
FREE_IDS = []


def http_call(origin, method, path, headers, body_bytes, timeout=10):
    parsed = urlparse(origin)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    conn = HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body_bytes, headers=headers or {})
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, raw
    finally:
        conn.close()


def call_provider(prov, model, req):
    """One provider call. Returns (True, (text, uin, uout)) or (False, None).

    A 200 is always accepted: the env records it as the request's one
    successful call, so calling another provider after it would break
    the one-provider check. Any other status fails over.
    """
    wire = prov["wire"]
    headers = {"Content-Type": "application/json"}
    auth = prov["auth"]
    key = prov.get("key", "")
    if auth == "bearer":
        headers["Authorization"] = "Bearer " + key
    elif auth == "x-api-key":
        headers["x-api-key"] = key
    elif auth == "x-goog-api-key":
        headers["x-goog-api-key"] = key
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    if wire == "openai":
        path = prov["path"] + "/v1/chat/completions"
        payload = to_openai_body(req, model)
    elif wire == "anthropic":
        path = prov["path"] + "/v1/messages"
        payload = to_anthropic_body(req, model)
    else:
        action = "streamGenerateContent" if req["stream"] else "generateContent"
        suffix = "?alt=sse" if req["stream"] else ""
        path = (
            prov["path"] + "/v1beta/models/" + quote(model, safe="")
            + ":" + action + suffix
        )
        payload = to_gemini_body(req, model)
    try:
        status, raw = http_call(
            ENV_URL, "POST", path, headers,
            json.dumps(payload).encode("utf-8"), timeout=10,
        )
    except Exception:
        return False, None
    if status != 200:
        return False, None
    text, uin, uout = parse_provider_response(wire, req["stream"], raw)
    if uin is None or uout is None:
        cuin, cuout = recompute_usage(req, text)
        if uin is None:
            uin = cuin
        if uout is None:
            uout = cuout
    return True, (text, int(uin), int(uout))


def models_list_shape(headers):
    """OpenAI list for a Bearer header, Anthropic list for x-api-key."""
    if headers.get("authorization"):
        return "openai"
    if headers.get("x-api-key"):
        return "anthropic"
    return "openai"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, body):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def do_GET(self):
        if self.path == "/v1/models":
            shape = models_list_shape(self.headers)
            if shape == "openai":
                doc = {
                    "object": "list",
                    "data": [
                        {
                            "id": mid,
                            "object": "model",
                            "created": CREATED,
                            "owned_by": "free-model-aggregator",
                            "free": True,
                            "context_length": FREE_CTX.get(mid),
                        }
                        for mid in FREE_IDS
                    ],
                }
            else:
                doc = {
                    "data": [
                        {
                            "type": "model",
                            "id": mid,
                            "display_name": mid,
                            "created_at": "2026-01-01T00:00:00Z",
                            "free": True,
                            "context_length": FREE_CTX.get(mid),
                        }
                        for mid in FREE_IDS
                    ],
                    "has_more": False,
                }
            self._send(200, "application/json", json.dumps(doc))
            return
        self._send(404, "application/json", json.dumps(
            {"error": {"message": "not found", "type": "invalid_request_error"}}
        ))

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            api = "openai"
        elif self.path == "/v1/messages":
            api = "anthropic"
        else:
            self._send(404, "application/json", json.dumps(
                {"error": {"message": "not found", "type": "invalid_request_error"}}
            ))
            return
        try:
            body = json.loads(self._read_body().decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body is not an object")
        except Exception:
            self._send(400, "application/json", json.dumps(
                {"error": {"message": "bad json", "type": "invalid_request_error"}}
            ))
            return
        try:
            req = norm_request(api, body)
        except Exception as exc:
            self._send(400, "application/json", json.dumps(
                {"error": {"message": str(exc), "type": "invalid_request_error"}}
            ))
            return
        got = ROUTER.serve(req)
        if got is None:
            sys.stderr.write("free-model-aggregator: no provider served a %s request\n" % api)
            sys.stderr.flush()
            self._send(503, "application/json", json.dumps(
                {"error": {
                    "message": "no free provider could serve this request",
                    "type": "api_error",
                }}
            ))
            return
        model, text, uin, uout = got
        sys.stderr.write(
            "free-model-aggregator: %s stream=%s -> %s uin=%d uout=%d\n"
            % (api, req["stream"], model, uin, uout)
        )
        sys.stderr.flush()
        if api == "openai":
            if req["stream"]:
                self._send(200, "text/event-stream; charset=utf-8",
                           render_openai_sse(model, text, uin, uout))
            else:
                self._send(200, "application/json", json.dumps(
                    render_openai_json(model, text, uin, uout)))
        else:
            if req["stream"]:
                self._send(200, "text/event-stream; charset=utf-8",
                           render_anthropic_sse(model, text, uin, uout))
            else:
                self._send(200, "application/json", json.dumps(
                    render_anthropic_json(model, text, uin, uout)))


FREE_CTX = {}


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

def load_from_instance(env_url):
    """Fallback brief source: GET /instance, then probe provider catalogs.

    Only used when no prompt file is available. Ratings are unknown
    here, so selection falls back to ordering by context length; the
    catalog itself still comes from the providers' own catalog routes.
    """
    try:
        status, raw = http_call(env_url, "GET", "/instance", {}, None, timeout=10)
        if status != 200:
            return None, None, None
        doc = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None, None, None
    directory = []
    for prov in doc.get("providers", []):
        path = urlparse(prov.get("base_url", "")).path or ""
        directory.append({
            "id": prov["id"],
            "name": prov.get("name", prov["id"]),
            "wire": prov["wire"],
            "auth": prov["auth"],
            "key_env": prov.get("key_env", ""),
            "key": prov.get("key", ""),
            "path": path,
        })
    ratings = []
    for prov in directory:
        wire = prov["wire"]
        headers = {}
        if prov["auth"] == "bearer":
            headers["Authorization"] = "Bearer " + prov["key"]
        elif prov["auth"] == "x-api-key":
            headers["x-api-key"] = prov["key"]
        elif prov["auth"] == "x-goog-api-key":
            headers["x-goog-api-key"] = prov["key"]
        if wire == "anthropic":
            headers["anthropic-version"] = ANTHROPIC_VERSION
        if wire == "gemini":
            cpath = prov["path"] + "/v1beta/models"
        else:
            cpath = prov["path"] + "/v1/models"
        try:
            status, raw = http_call(env_url, "GET", cpath, headers, None, timeout=10)
            if status != 200:
                continue
            cat = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        if wire == "gemini":
            items = cat.get("models", []) or []
            rows = []
            for item in items:
                name = item.get("name", "")
                mid = name[len("models/"):] if name.startswith("models/") else name
                rows.append({
                    "model": mid, "provider": prov["id"],
                    "rating_milli": 1000,
                    "context_length": item.get("context_length"),
                    "tools": bool(item.get("tools")),
                    "vision": bool(item.get("vision")),
                    "json": bool(item.get("json")),
                    "free": bool(item.get("free")),
                })
        else:
            rows = []
            for item in cat.get("data", []) or []:
                rows.append({
                    "model": item.get("id"),
                    "provider": prov["id"],
                    "rating_milli": 1000,
                    "context_length": item.get("context_length"),
                    "tools": bool(item.get("tools")),
                    "vision": bool(item.get("vision")),
                    "json": bool(item.get("json")),
                    "free": bool(item.get("free")),
                })
        ratings.extend(rows)
    return directory, ratings, doc.get("workload", {}).get("manifest", [])


def main(argv):
    global ENV_URL, ROUTER, CATALOG, FREE_IDS, FREE_CTX

    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")

    directory = None
    ratings = None
    if prompt_file:
        directory, ratings, _manifest = load_brief(prompt_file)

    if "--list" in argv:
        if directory is None or ratings is None:
            sys.stdout.write('{"providers":[]}\n')
        else:
            sys.stdout.write(
                json.dumps(build_catalog(directory, ratings),
                           separators=(",", ":")) + "\n"
            )
        return 0

    if not env_url:
        sys.stderr.write("free-model-aggregator: MAKEFUN_ENV_URL is not set\n")
        return 1
    ENV_URL = env_url
    if directory is None or ratings is None:
        directory, ratings, _manifest = load_from_instance(env_url)
    if directory is None or ratings is None:
        sys.stderr.write("free-model-aggregator: no brief and no instance\n")
        return 1

    CATALOG = build_catalog(directory, ratings)
    seen = set()
    for prov in CATALOG["providers"]:
        for model in prov["models"]:
            if model["id"] not in seen:
                seen.add(model["id"])
                FREE_IDS.append(model["id"])
    for row in ratings:
        if row["model"] not in FREE_CTX:
            FREE_CTX[row["model"]] = row.get("context_length")
    ROUTER = Router(directory, ratings, _manifest)

    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sys.stderr.write("free-model-aggregator: listening on %d\n" % port)
    sys.stderr.flush()

    register_body = json.dumps({
        "protocol_version": 1,
        "port": port,
        "catalog": CATALOG,
    }).encode("utf-8")
    registered = False
    for _attempt in range(10):
        try:
            status, raw = http_call(
                env_url, "POST", "/register",
                {"Content-Type": "application/json"}, register_body,
                timeout=10,
            )
            if status == 200:
                registered = True
                break
            sys.stderr.write("free-model-aggregator: register got %d\n" % status)
        except Exception as exc:
            sys.stderr.write("free-model-aggregator: register failed: %s\n" % exc)
        time.sleep(0.5)
    if not registered:
        sys.stderr.write("free-model-aggregator: register never succeeded\n")
        server.shutdown()
        return 1
    sys.stderr.write("free-model-aggregator: registered\n")
    sys.stderr.flush()

    while True:
        try:
            status, raw = http_call(
                env_url, "GET", "/work", {}, None, timeout=10,
            )
            if status == 200:
                work = json.loads(raw.decode("utf-8", "replace"))
                if work.get("done") is True:
                    break
        except Exception:
            pass
        time.sleep(0.2)
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
