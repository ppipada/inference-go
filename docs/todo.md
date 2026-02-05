# Project TODO

- [Input/output modalities](#inputoutput-modalities)
  - [Done IO](#done-io)
  - [TODO IO](#todo-io)
  - [Deferred IO](#deferred-io)
- [Top-level request params \& controls](#top-level-request-params--controls)
  - [Done Params](#done-params)
  - [Deferred Params](#deferred-params)
- [Tools](#tools)
  - [Done Tools](#done-tools)
  - [Deferred Tools](#deferred-tools)
- [Output metadata](#output-metadata)
  - [Done Output](#done-output)
  - [TODO Output](#todo-output)
  - [Deferred Output](#deferred-output)
- [Context management](#context-management)
  - [TODO Context](#todo-context)
  - [Deferred Context](#deferred-context)

## Input/output modalities

### Done IO

- Conversation turns / input messages
  - Normalized: `FetchCompletionRequest.Messages`
  - Anthropic: top-level `messages` (role = user/assistant; system is separate top-level `system`)
  - OpenAI Responses: top-level `input` (string or array; adapter uses array-of-{role,content} form)
  - OpenAI Chat: top-level `messages`
  - Notes
    - OpenAI Chat message objects may include `name` (not currently normalized; see Deferred Params)

- Attachments (stateless approach; no vendor file IDs)
  - Normalized: `ContentItemImage`, `ContentItemFile`
  - Anthropic
    - Images supported (base64 or URL)
    - Documents supported for PDFs (base64 or URL)
    - Text file support is intended and tracked (see Context management TODO: “Anthropic plain-text document support”)
  - OpenAI Responses
    - Images and files supported (base64 or URL)
    - Input/output must not rely on vendor `file_id` (stateful is not supported)
  - OpenAI Chat
    - Images supported (base64 or URL)
    - Files supported (base64 only)
  - Notes (carried forward)
    - Input message: everything supported except stateful properties inside image/file content (notably `file_id`)
    - Tool call outputs: everything supported except stateful properties (notably `file_id`)

### TODO IO

- Citations beyond URL (stateless subset only)
  - Spec change
    - Optionally extend `spec.CitationKind` for stateless offsets (page/char), but do not add vendor file handles
  - Priority: Anthropic P2, Responses P2, Chat P2
  - Notes (carried forward)
    - OpenAI Responses: do not support citations like file/container/filepath in stateless mode; only normalize stateless representations
    - Anthropic: do not support citation variants like content block location/search result location for now; keep URL citations only unless a stateless design is clear

### Deferred IO

- Image output modality
  - Anthropic does not support image generation
  - OpenAI supports image output via an image generation tool, not as standard message content
  - Google generate content may treat image gen as direct I/O content via dedicated models
  - Given the mismatch, keep image output as deferred until a deliberate cross-vendor abstraction is chosen

- Cross-provider:
  - vendor file IDs and stateful file ecosystems (not supported)
  - No reliance on vendor `file_id` in inputs/outputs

- Item reference: unclear; likely stateful; explicitly deferred
- Audio
- Anthropic:
  - SearchResultBlock support (input or output); deferred
  - Document “content block source” inside document source; deferred (unclear when to use vs top-level text/image)
    - Note: why content block source exists inside document source for input; unclear when it should be used vs top-level text/image.
  - Citation types beyond URL; deferred
    - char location
    - page location
    - content block location
    - search result location

## Top-level request params & controls

### Done Params

- Model selection
  - Normalized: `ModelParam.Name`
  - Anthropic: `model`
  - OpenAI Responses: `model`
  - OpenAI Chat: `model`

- System/developer instructions
  - Normalized: `ModelParam.SystemPrompt`
  - Anthropic: top-level `system`
  - OpenAI Responses: `instructions`
  - OpenAI Chat: `system`/`developer` message (adapter policy)

- Max output tokens
  - Normalized: `ModelParam.MaxOutputLength`
  - Anthropic: `max_tokens`
  - OpenAI Responses: `max_output_tokens`
  - OpenAI Chat: `max_tokens`

- Temperature
  - Normalized: `ModelParam.Temperature`
  - Anthropic: `temperature`
    - Adapter rule: when thinking is enabled, temperature is not allowed (enforce provider constraint)
  - OpenAI Responses: `temperature`
  - OpenAI Chat: `temperature`

- Streaming
  - Normalized: `ModelParam.Stream` + `FetchCompletionOptions.StreamHandler`
  - Anthropic: `stream`
  - OpenAI Responses: `stream`
  - OpenAI Chat: `stream`
  - Notes
    - Unified stream events for text and thinking where exposed
    - OpenAI Responses stream-only options like `include_obfuscation` are not implemented (tracked as deferred under “Vendor-specific controls” below)

- Reasoning / thinking
  - Normalized: `ModelParam.Reasoning` + `ReasoningContent`
  - Anthropic: thinking blocks, redacted thinking supported
  - OpenAI Responses: reasoning config plus reasoning items supported
    - Note: reasoning summary/verbosity controls are pending (tracked in TODO)
  - OpenAI Chat: `reasoning_effort` supported as config only
    - Chat does not support reasoning blocks as message content; adapter drops reasoning messages

- Structured output (output format / JSON schema)
  - Normalized: `ModelParam.OutputFormat *spec.OutputFormat`
  - OutputFormat minimal shape
    - `type = text`
    - `type = json_schema` with `schema` object, optional `strict` bool, optional `name` string
  - Anthropic mapping
    - `output_config.format` for `json_schema`

  - OpenAI Responses mapping
    - `text.format` with `type = text/json_schema` (plus `json_schema` payload)
  - OpenAI Chat mapping
    - `response_format` with `type = text/json_schema` (plus `json_schema` payload)

- Text generated verbosity control:
  - `Verbosity *string`
  - In OpenAI responses

- Tool selection policy (separate from tool definitions)
  - Normalized
    - `FetchCompletionRequest.ToolPolicy *spec.ToolPolicy`
    - `ToolChoices` as tool definitions available
  - Anthropic: `tool_choice` (auto/any/tool/none patterns)
  - OpenAI Responses: `tool_choice` (string/object)
  - OpenAI Chat: `tool_choice`

- Disable parallel tool use (at most one tool call)
  - Normalized: `ToolPolicy.DisableParallel bool`
  - Anthropic: `tool_choice.*.disable_parallel_tool_use`
  - OpenAI Chat: `parallel_tool_calls`

- Stop sequences
  - Normalized: `ModelParam.StopSequences []string`
  - Anthropic: `stop_sequences`
  - OpenAI Chat: `stop` (string or array)
  - OpenAI Responses: wrapper-level unsupported for now (omit; no client-side trimming)

- Reasoning summary control
  - Normalized: extend `ReasoningParam` with `SummaryStyle *string`
  - OpenAI Responses: summary-related config only
  - Anthropic: no direct equivalent, no-op
  - OpenAI Chat: no direct equivalent beyond `reasoning_effort`, no-op

### Deferred Params

- Guardrails for any new top-level params
  - Constraints
    - Must be allowlisted
    - Must not introduce stateful behavior
    - Must fail closed on unknown keys
  - Note: “P2 can be considered as deferred”

- Tool options
  - Max tool calls
  - Notes
    - Prefer wrapper-enforced cap (reject/ignore extra tool calls) unless a provider-native control exists; must remain stateless

- Safe provider passthrough (allowlisted merge)
  - Spec change
    - Implement `ModelParam.AdditionalParametersRawJSON` merged into vendor request with per-adapter allowlist and validation
  - Priority: Anthropic P1, Responses P1, Chat P1
  - Notes
    - Old matrix explicitly said metadata/service tiers not supported; passthrough is the only safe place they could be exposed later
    - Even with passthrough implemented, sensitive/vendor-state-adjacent knobs remain deferred unless explicitly allowlisted

- Top-p
  - Spec change: add `ModelParam.TopP *float64`
  - Anthropic: `top_p`
  - OpenAI Responses: `top_p`
  - OpenAI Chat: `top_p`
  - Priority: Anthropic P2, Responses P2, Chat P2
  - Validation: range `0..1`

- Logprobs
  - Spec change: add `ModelParam.LogProbs` config and output fields
  - OpenAI Responses: potential mapping `top_logprobs` plus include paths if supported
  - OpenAI Chat: logprobs controls model-dependent
  - Anthropic: no equivalent
  - Priority: Responses P2, Chat P2

- Presence and frequency penalties
  - Spec change: add `ModelParam.PresencePenalty`, `ModelParam.FrequencyPenalty`
  - OpenAI Chat: `presence_penalty`, `frequency_penalty`
  - OpenAI Responses: only if supported in target API; otherwise ignore
  - Anthropic: no equivalent
  - Priority: Responses P2, Chat P2

- Seed and logit_bias
  - Spec change: prefer allowlisted passthrough rather than first-class fields (unless required to normalize)
  - Priority: Responses P2, Chat P2

- Stateful params and server-managed continuation controls (not supported)
  - `background`
  - `conversation`
  - prompt objects
  - `previous_response_id`

- Anthropic vendor-specific controls (stateless only)
  - `metadata.user_id` (safety/abuse tracking)
  - `service_tier` (`auto` / `standard_only`)
  - `top_k`

- OpenAI Chat vendor-specific controls (stateless only)
  - `prompt_cache_key` and `prompt_cache_retention`
  - `safety_identifier`
  - `service_tier`
  - Message field `name` (not normalized; ignore or allowlisted passthrough only)

- OpenAI Responses vendor-specific controls (stateless only)
  - `include` (controls extra output items)
  - `store` (provider-side storage toggle)
  - `stream_options`
    - `include_obfuscation` in stream options
  - `prompt_cache_key` and `prompt_cache_retention`
  - `safety_identifier` / `userid` for safety tracking
  - `service_tier`
  - `metadata` (opaque kv map)
    - If ever needed: only via allowlisted passthrough and still subject to stateless constraints
  - `truncation`
  - Text verbosity controls outside of reasoning (do not standardize), e.g. `text.verbosity`

## Tools

### Done Tools

- Tools (client function/custom tool definitions)
  - Normalized: `FetchCompletionRequest.ToolChoices` currently treated as “tool definitions available”
  - Anthropic: `tools[]`
  - OpenAI Responses: `tools[]`
  - OpenAI Chat: `tools[]`
  - Notes
    - Function call and custom tool call content are supported in input and output
    - Tool selection behavior not normalized yet (see TODO: `ToolPolicy`)

- Web search
  - Normalized: `ToolTypeWebSearch` tool def/choice mapping
  - Anthropic: server web search tool use supported
  - OpenAI Responses: built-in web search tool use supported
  - OpenAI Chat: `web_search_options` supported
  - Notes
    - Output shape differs by provider; OpenAI often surfaces results as citations/annotations
    - Note: server tool use block with websearch input typed as `any` is odd; watch schema stability and validation strategy

### Deferred Tools

- Tool options
  - Max tool calls (tracked under Top-level params)

- Cross-provider (explicitly not doing / out of scope)
  - Bash/Shell: Local Tool available.
  - Patch/Text editor tool: Local Tool available.
  - Computer use
  - Code execution (hosted sandbox) / code interpreter families

- OpenAI Responses specific:
  - Image generation call
  - Local shell tool (outdated; codex mini only)
  - MCP-specific schemas
    - MCP list tools
    - Approval request/response
    - MCP tool call
  - File search (vector stores)
  - Remote MCP
  - Remote connectors

- Anthropic: tool choice ecosystem deferrals
  - Remote MCP connector
  - Programmatic tool calling (looped code execution plus tool calls)
  - Memory tool
  - Tool search tool (server-side tool library and search)
  - Web fetch tool
    - Not doing because local URL fetch and send is better for stateless control, processing, and error handling
  - Note: server tool use block with websearch input typed as `any` is odd; keep an eye on schema stability and validation strategy.

## Output metadata

### Done Output

- Usage
  - Normalized: `Usage`
  - Supported usage: input tokens, output tokens, cached tokens usage
  - Supported usage: input/output tokens usage, plus cached token accounting where exposed

- Supported error surfacing

- HTTP debugging - `DebugDetails`
  - Pluggable `CompletionDebugger` interface (span-based)
  - Built-in `debugclient.HTTPCompletionDebugger`
  - Scrubbed HTTP request/response metadata attached to `FetchCompletionResponse.DebugDetails`

- Output params not explicitly decodd are present as opaque debug/details payload.

### TODO Output

- Promote more response metadata (small stable set)
  - Extend `FetchCompletionResponse` with minimal stable set like `id`, `model`, `stop_reason`, `stop_sequence` when safe
  - Priority: Anthropic P2, Responses P2, Chat P2
  - Notes
    - Old matrix: these were not promoted and remained opaque details/debug

### Deferred Output

- Provider metadata kv/service tiers/etc
  - Deferred unless introduced via safe allowlisted passthrough (see Top-level “Safe provider passthrough”)

- OpenAI Responses output params not promoted
  - Everything except usage (including cached token usage) and error remains opaque details/debug

## Context management

- Heuristic prompt filtering by approximate token count
  - Normalized: `ModelParam.MaxPromptLength`
  - Implementation: `sdkutil.FilterMessagesByTokenCount` (approximate / heuristic)
  - Notes
    - This is not a provider-native truncation control; it is wrapper-side filtering to keep requests under a target size.

### TODO Context

- CacheControl mapping (ephemeral)
  - Spec change: none (`spec.CacheControl` exists)
  - Anthropic: `cache_control` on supported blocks (text/image/document/tool)
  - OpenAI Responses: ignore (no direct equivalent)
  - OpenAI Chat: ignore (no direct equivalent)
  - Priority: Anthropic P0, Responses P2, Chat P2

- Anthropic plain-text document support
  - Spec change: none (already `ContentItemFile.FileMIME` / `ContentItemFile.FileData`)
  - Implementation
    - Decode base64 for `text/*` and map to Anthropic plain-text document source
  - Priority: Anthropic P0
  - Notes (carried forward)
    - Old matrix: “Document source supports pdf, url pdf and text file”; this ensures wrapper maps text files rather than ignoring them

### Deferred Context

- Stateful conversation / continuation semantics
  - Covered under Top-level deferred: `background`, `conversation`, prompt objects, `previous_response_id`
- Compaction: stateful; explicitly deferred
