# LLM Inference for Go

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Go Report Card](https://goreportcard.com/badge/github.com/flexigpt/inference-go)](https://goreportcard.com/report/github.com/flexigpt/inference-go)
[![lint](https://github.com/flexigpt/inference-go/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/flexigpt/inference-go/actions/workflows/lint.yml)
[![test](https://github.com/ppipada/inference-go/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/ppipada/inference-go/actions/workflows/test.yml)

A single interface in Go to get inference from multiple LLM / AI providers using their official SDKs.

- [Features at a glance](#features-at-a-glance)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Examples](#examples)
- [Supported providers](#supported-providers)
  - [Anthropic Messages API](#anthropic-messages-api)
  - [OpenAI Responses API](#openai-responses-api)
  - [OpenAI Chat Completions API](#openai-chat-completions-api)
- [HTTP debugging](#http-debugging)
- [Notes](#notes)
- [Development](#development)
- [License](#license)

## Features at a glance

- Single normalized interface (`ProviderSetAPI`) for multiple providers. Current support:
  - Anthropic Messages API. [Official SDK used](https://github.com/anthropics/anthropic-sdk-go)
  - OpenAI Chat Completions API [Official SDK used](https://github.com/openai/openai-go)
  - OpenAI Responses API [Official SDK used](https://github.com/openai/openai-go)

- Normalized data model in `spec/`:
  - messages (user / assistant / system/developer instructions are provided via `ModelParam.SystemPrompt`),
  - text, images, and files, (no audio/video content types yet),
  - tools (function, custom, built-in tools like web search),
  - reasoning / thinking content,
  - streaming events (text + thinking),
  - usage accounting.

- Streaming support:
  - Text streaming for all providers that support it.
  - Reasoning / thinking streaming where the provider exposes it (Anthropic, OpenAI Responses).

- Client and Server Tools:
  - Client tools are supported via Function Calling.
  - Anthropic server-side web search.
  - OpenAI Responses web search tool.
  - OpenAI Chat Completions web search via `web_search_options`.

- HTTP-level debugging:
  - Pluggable `CompletionDebugger` interface.
  - A built-in ready to use implementation at: `debugclient.HTTPCompletionDebugger`:
    - wraps SDK HTTP clients,
    - captures request/response metadata,
    - redacts secrets and sensitive content,
    - attaches a scrubbed debug blob to `FetchCompletionResponse.DebugDetails`.

## Installation

```bash
# Go 1.25+
go get github.com/flexigpt/inference-go
```

## Quickstart

Basic pattern:

1. Create a `ProviderSetAPI`.
2. Add one or more providers. Set their API keys.
3. Send a `FetchCompletionRequest`.

## Examples

- [Basic OpenAI Responses](./internal/integration/openai_responses_basic_example_test.go)
- [Basic OpenAI Chat Completions](./internal/integration/openai_chat_example_test.go)
- [Basic Anthropic Messages](./internal/integration/anthropic_example_test.go)
- [Extended OpenAI Responses example](./internal/integration/openai_responses_tools_attachments_example_test.go)
  - Demonstrates tools, web search, file and image attachments.

## Supported providers

### Anthropic Messages API

Feature support

| Area                      | Supported? | Notes                                                                                                        |
| ------------------------- | ---------: | ------------------------------------------------------------------------------------------------------------ |
| Text input/output         |        yes | User and assistant messages mapped to text blocks.                                                           |
| Streaming text            |        yes |                                                                                                              |
| Reasoning / thinking      |        yes | Thinking/Redacted is supported; redacted is not streamed to caller. Thinking enabled == temperature omitted. |
| Streaming thinking        |        yes |                                                                                                              |
| Images (input)            |        yes | Inline base64 (`imageData`) or remote URLs (`imageURL`) mapped to Anthropic image blocks.                    |
| Files / documents (input) |        yes | PDFs only, via base64 or URL. Plain-text base64 and other MIME types are currently ignored.                  |
| Audio/Video input/output  |         no |                                                                                                              |
| Tools (function/custom)   |        yes | JSON Schema based.                                                                                           |
| Web search                |        yes | Server web search tool use + web search tool-result blocks.                                                  |
| Citations                 |    partial | URL citations only. Other stateful citations are not mapped.                                                 |
| Metadata / service tiers  |     opaque | Not exposed in normalized types; available in debug payload.                                                 |
| Stateful flows            |         no | Library focuses on stateless calls only.                                                                     |
| Usage data                |        yes | Input/Output/Cached. Anthropic doesn't expose Reasoning tokens usage.                                        |

- Behavior for conversational + interleaved reasoning message input
  - Input: No reasoning content in the incoming messages.
    - Action: Build the message list unchanged. If the last user message is a `tool_result`, force _thinking disabled_; otherwise, honor the requested thinking setting.
  - Input: All reasoning messages are signed.
    - Action: Build the message list unchanged. If the last user message is a `tool_result` _and_ the previous assistant message begins with thinking content, force _thinking enabled_; otherwise, honor the requested thinking setting.
  - Input: Mix of reasoning messages where some include a valid signature thinking and others do not.
    - Action: Retain only the reasoning messages with a valid signature; drop the rest. Apply the above behaviors after this cleanup.

### OpenAI Responses API

Feature support

| Area                      | Supported? | Notes                                                                                                              |
| ------------------------- | ---------: | ------------------------------------------------------------------------------------------------------------------ |
| Text input/output         |        yes | Input/output messages fully supported.                                                                             |
| Streaming text            |        yes |                                                                                                                    |
| Reasoning / thinking      |        yes | Reasoning outputs are mapped. Reasoning **inputs** are accepted only as `encrypted_content`; others are dropped.   |
| Streaming thinking        |        yes |                                                                                                                    |
| Images (input)            |        yes | `imageData` (base64) or `imageURL`, with `detail` low/high/auto, mapped to Responses `input_image` items.          |
| Files / documents (input) |        yes | `fileData` (base64) or `fileURL` mapped to Responses `input_file` items; works for PDFs and other file MIME types. |
| Audio/Video input/output  |         no |                                                                                                                    |
| Tools (function/custom)   |        yes | JSON Schema based. Note: `custom` tool **definitions** are currently emitted as `function` tools.                  |
| Web search                |        yes | Calls are mapped when emitted; results typically surface as citations/annotations in text.                         |
| Citations                 |        yes | URL citations mapped to `spec.CitationKindURL`.                                                                    |
| Metadata / service tiers  |     opaque | Not exposed in normalized types; available in debug payload.                                                       |
| Stateful flows            |         no | Store is explicitly disabled (`Store: false`).                                                                     |
| Usage data                |        yes | Input/Output/Cached/Reasoning.                                                                                     |

- Behavior for conversational + interleaved reasoning message input
  - Input: No reasoning messages.
    - Action: Build the message list unchanged. Honor the requested thinking setting.
  - Input: All reasoning messages are `encrypted_content`.
    - Action: Build the message list unchanged. Honor the requested thinking setting.
  - Input: Mixed reasoning messages: some are signature-based and some are `encrypted_content`.
    - Action: Keep only the `encrypted_content` reasoning; drop the signature-based reasoning.

### OpenAI Chat Completions API

Feature support

| Area                      | Supported? | Notes                                                                                                             |
| ------------------------- | ---------: | ----------------------------------------------------------------------------------------------------------------- |
| Text input/output         |        yes | Only the first choice from output is surfaced up.                                                                 |
| Streaming text            |        yes |                                                                                                                   |
| Reasoning / thinking      |        yes | Reasoning effort config only; no separate reasoning messages in API.                                              |
| Streaming thinking        |         no | Not exposed by Chat Completions.                                                                                  |
| Images (input)            |        yes | `imageData` (base64) and `imageURL` are both supported; base64 is sent as a data URL with `detail` low/high/auto. |
| Files / documents (input) |        yes | `fileData` (base64) only, sent as a data URL; `fileURL` and stateful file IDs are not used by this adapter.       |
| Audio/Video input/output  |         no |                                                                                                                   |
| Tools (function/custom)   |        yes | JSON Schema based. Note: `custom` tool **definitions** are currently emitted as `function` tools.                 |
| Web search                |        yes | API doesn't expose a tool; mapped via top-level `web_search_options` derived from a `webSearch` ToolChoice.       |
| Citations                 |        yes | URL citations mapped from annotations.                                                                            |
| Metadata / service tiers  |     opaque | Not exposed in normalized types; available in debug payload.                                                      |
| Stateful flows            |         no | Library focuses on stateless calls only.                                                                          |
| Usage data                |        yes | Input/Output/Cached/Reasoning.                                                                                    |

- Behavior for conversational + interleaved reasoning message input
  - Reasoning effort config is kept as is.
  - All reasoning input/output messages are dropped as the api doesn't support it.

## HTTP debugging

The library exposes a pluggable `CompletionDebugger` interface:

```go
type CompletionDebugger interface {
    HTTPClient(base *http.Client) *http.Client
    StartSpan(ctx context.Context, info *spec.CompletionSpanStart) (context.Context, spec.CompletionSpan)
}
```

- package `debugclient` includes an implementation that can be readily used as `HTTPCompletionDebugger`:
  - wraps the provider SDK’s `*http.Client`,
  - captures and scrubs:
    - URL, method, headers (with secret redaction),
    - query params,
    - request/response bodies (optional, scrubbed of LLM text and large base64),
    - curl command for reproduction,
  - attaches a structured `HTTPDebugState` to `FetchCompletionResponse.DebugDetails`.
  - You can then inspect `resp.DebugDetails` for a given call, or just rely on `slog` output.

- Use it via `WithDebugClientBuilder`:

```go
ps, _ := inference.NewProviderSetAPI(
    inference.WithDebugClientBuilder(func(p spec.ProviderParam) spec.CompletionDebugger {
        return debugclient.NewHTTPCompletionDebugger(&debugclient.DebugConfig{
            LogToSlog: false,
        })
    }),
)
```

## Notes

- Stateless focus. The design focuses on stateless request/response interactions:
  - no conversation IDs,
  - no file IDs,

- Opaque / provider‑specific fields.
  - Many provider‑specific fields (error details, service tiers, cache metadata, full raw responses) are only available through the debug payload, not in the normalized `spec` types.
  - Few of the common needed params may be added over time and as needed.

- Token counting - Normalized `Usage` reports what the provider exposes:
  - Anthropic: input vs. cached tokens, output tokens.
  - OpenAI: prompt vs. cached tokens, completion tokens, reasoning tokens where available.

- Heuristic prompt filtering.
  - `ModelParam.MaxPromptLength` triggers `sdkutil.FilterMessagesByTokenCount`, which uses a simple heuristic token counter. It is approximate, not an exact tokenizer.

## Development

- Formatting follows `gofumpt` and `golines` via `golangci-lint`, which is also used for linting. All rules are in [.golangci.yml](.golangci.yml).
- Useful scripts are defined in `taskfile.yml`; requires [Task](https://taskfile.dev/).
- Bug reports and PRs are welcome:
  - Keep the public API (`package inference` and `spec`) small and intentional.
  - Avoid leaking provider‑specific types through the public surface; put them under `internal/`.
  - Please run tests and linters before sending a PR.

## License

Copyright (c) 2026 - Present - Pankaj Pipada

All source code in this repository, unless otherwise noted, is licensed under the MIT License.
See [LICENSE](./LICENSE) for details.
