# File Input Guide

This guide covers how to send files (PDFs, audio, video, images, and other documents) to the Gemini model through `GeminiClient`.

---

## Table of Contents

1. [Overview](#overview)
2. [The Two File Types](#the-two-file-types)
3. [Method Comparison](#method-comparison)
4. [Inline Files with `File`](#inline-files-with-file)
5. [Uploaded Files with `upload_file()`](#uploaded-files-with-upload_file)
6. [Passing Files to Chat Methods](#passing-files-to-chat-methods)
7. [Mixing Files in One Request](#mixing-files-in-one-request)
8. [Common MIME Types](#common-mime-types)

---

## Overview

Gemini natively understands multimodal input. Syndicate exposes this through two value objects and the `files=` parameter on both `chat_completion_async` and `chat_completion_stream`.

```python
from syndicate import File
from syndicate.clients.gemini import GeminiClient, UploadedFile
```

Files are always attached to the **last user message** in the conversation. There is no separate file-only turn — the file and the prompt travel together.

---

## The Two File Types

### `File` — inline bytes

```python
File(data: bytes, mime_type: str, name: Optional[str] = None)
```

The raw bytes are serialized directly in the request payload. Use this for transient data where you don't need to reuse the file across requests.

- **`data`** — required, must be `bytes` or `bytearray`. Strings are rejected.
- **`mime_type`** — required, must be set correctly or the model will misinterpret the content.
- **`name`** — optional display name, used for logging/UI only.

```python
with open("report.pdf", "rb") as f:
    file = File(data=f.read(), mime_type="application/pdf")

# or with pathlib
from pathlib import Path
file = File(data=Path("report.pdf").read_bytes(), mime_type="application/pdf")
```

### `UploadedFile` — Gemini server-side reference

```python
# Returned by GeminiClient.upload_file() — do not construct directly
UploadedFile(uri: str, mime_type: str, name: str)
```

A reference to a file stored server-side via the Gemini File API. This type lives in
`syndicate.clients.gemini` because it is Gemini-specific — other providers will define
their own equivalent when they add upload support.

You never construct `UploadedFile` yourself; it is always returned by `upload_file()`.

---

## Method Comparison

| | `File` (inline) | `UploadedFile` (File API) |
|---|---|---|
| Max size | 100 MB (50 MB for PDFs) | 2 GB per file |
| Bytes sent per request | Yes — every time | No — only URI sent |
| Best for | One-off requests, small files | Large files, reuse across multiple calls |
| Requires upload step | No | Yes — `await client.upload_file(file)` |
| Server-side TTL | None | 48 hours |

---

## Inline Files with `File`

The simplest path: read bytes, wrap in `File`, pass to the chat method.

```python
from pathlib import Path
from syndicate import File, Message
from syndicate.clients.gemini import GeminiClient

client = GeminiClient(model_name="gemini-2.5-pro", api_key="...")

pdf_bytes = Path("contract.pdf").read_bytes()

response = await client.chat_completion_async(
    messages=[Message(role="human", content="Summarize the key terms in this contract.")],
    files=[File(data=pdf_bytes, mime_type="application/pdf")],
)
print(response.content)
```

### With an agent

`**kwargs` passed to `agent.invoke()` are forwarded to the underlying client, so `files=` works directly:

```python
from pathlib import Path
from syndicate import File, GenericAgent
from syndicate.clients import GeminiClient

client = GeminiClient(model_name="gemini-2.5-pro", api_key="...")
agent = GenericAgent(
    llm_client=client,
    system_prompt="You are a document analyst.",
)

pdf_bytes = Path("contract.pdf").read_bytes()

response = await agent.invoke(
    "Summarize the key terms in this contract.",
    files=[File(data=pdf_bytes, mime_type="application/pdf")],
)
print(response)
```

---

## Uploaded Files with `upload_file()`

Use this when the same file will be referenced in multiple requests, or when the file exceeds 100 MB.

```python
from pathlib import Path
from syndicate import File, Message
from syndicate.clients.gemini import GeminiClient, UploadedFile

client = GeminiClient(model_name="gemini-2.5-pro", api_key="...")

# 1. Upload once
audio_bytes = Path("interview.mp3").read_bytes()
uploaded = await client.upload_file(File(data=audio_bytes, mime_type="audio/mp3", name="interview.mp3"))
# uploaded is an UploadedFile with .uri, .mime_type, .name

# 2. Reuse across multiple requests within the 48h window
summary = await client.chat_completion_async(
    messages=[Message(role="human", content="Summarize this audio.")],
    files=[uploaded],
)

transcript = await client.chat_completion_async(
    messages=[Message(role="human", content="Provide a full transcript.")],
    files=[uploaded],  # same upload, no re-transfer
)
```

`upload_file()` accepts a `File` and returns an `UploadedFile`. The `File.name` field becomes the `display_name` in the File API if provided.

---

## Passing Files to Chat Methods

Both `chat_completion_async` and `chat_completion_stream` accept the same `files=` parameter.

```python
# Non-streaming
response = await client.chat_completion_async(
    messages=[Message(role="human", content="What is in this image?")],
    files=[File(data=img_bytes, mime_type="image/png")],
)

# Streaming
async for chunk in client.chat_completion_stream(
    messages=[Message(role="human", content="Describe this video.")],
    files=[uploaded_video],
):
    print(chunk.content, end="", flush=True)
```

Note: `chat_completion_stream` is an async generator — iterate it directly without `await`.

---

## Mixing Files in One Request

You can send multiple files and mix `File` and `UploadedFile` in the same request:

```python
response = await client.chat_completion_async(
    messages=[Message(role="human", content="Compare these two documents.")],
    files=[
        File(data=Path("doc_a.pdf").read_bytes(), mime_type="application/pdf"),
        uploaded_doc_b,   # UploadedFile from a previous upload_file() call
    ],
)
```

Files are appended to the last user message's parts in the order they appear in the list.

---

## Common MIME Types

| Content | MIME type |
|---|---|
| PDF | `application/pdf` |
| Plain text | `text/plain` |
| PNG image | `image/png` |
| JPEG image | `image/jpeg` |
| MP3 audio | `audio/mp3` |
| MP4 video | `video/mp4` |
| CSV | `text/csv` |
| HTML | `text/html` |

Always set the MIME type accurately. Gemini uses it to select the correct parser for the file content.
