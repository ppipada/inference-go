# Notes

## Support Matrix Notes: Anthropic

- Input params

  - Supported: Max tokens, model, stream, temperature, thinking
  - Not supported:
    - metadata : userid - to detect safety issues.
    - service tiers
    - stop sequences array
    - topK, topP

- OutputParams

  - Supported: input output tokens usage
  - Not supported: All things not handled are passed as opaque Details.
    - id, model, stop reason, stop sequence, cache tokens, service tier

- Input output content (i.e messages and system prompt):

  - Text:
    - Support: Text input, URL citations i.e web search result.
    - Don't support citations: char location, page location, content block location, search result location.
      - The usecase for a few is known, but need to distinguish which one is useful for actual stateless api operation, vs which is strictly for stateful flows.
  - Image: all supported.
  - Document source:
    - Support: pdf, url pdf and text file.
    - Don't support: Content block source
      - There is a content block as a option inside the possible document blocks
      - It allows to send text and and image as document param.
      - Not sure why this duplication from the top level param is present and when is this supposed to be used vs top level.
  - Thinking, redacted thinking: all supported.

  - Tool use: all supported including Server tool use of websearch.
  - Tool result:

    - Support: Text, image, document, Web search tool result.
    - Don't support: Similar to above, search result block and content block source inside document source is not supported in the result type.

  - SearchResultBlock - Fully not supported in input or output.
    - Need to see where it is useful in stateless flows.

- Tool choice:

  - [ ] Deferred

    - [ ] Remote MCP connector: decide when to do mcp integrations.

  - [ ] Deferred/Think through

    - [ ] Bash tool: Similar to OpenAIs shell tool. Both recommend handing things in a "session".
    - [ ] Text editor: This is near to patch apply tool in behavior but has some text like commands string replace and view etc.
    - [ ] Web fetch: it is page fetch for content or pdf. For normal usecase the current local url fetch and send will be better in terms of state management and better processing and errors. This may be useful when doing web search integration. think about it when doing that.

  - [ ] Don't

    - [ ] Computer use: Same as openai.
    - [ ] Code execution: Same as openai code interpreter with some free hours of usage.
    - [ ] Programmatic tool calling: this is a convoluted looping of code execution and local tool calls. No usecase known yet.
      - [ ] Simple mental model: Let Claude write a little Python script once, and run that script in a loop while it calls your tools, instead of asking Claude to think and call tools over and over.
      - [ ] May be useful only to save tokens when there is a looped tool call need.
    - [ ] Memory tool: This seems interesting on first pass. it is similar to text editor in flow, but allows to accumulate context locally on client. Would be interesting to see how to do context management in local app ourselves and how this tool helps with that?
    - [ ] Tool search tool: This is a server side tool that allows claude to hold a tool library on its server and then search through it to get appropriate tool and then invoke it. It is very efficient in terms of token usage and can be checked on how to implement it locally.

## Support Matrix Notes: OpenAI Responses

- Input params

  - Supported: Max ouput tokens, model, stream, temperature, reasoning (support reasoning summary config is pending), Instructions
  - Not supported:

    - Tool options: max tool calls, parallel_tool_calls
    - metadata : opaque kv pairs.
    - prompt_cache_key, prompt_cache_retention
    - safety_identifier/userid - to detect safety issues.
    - include_obfuscation in stream options
    - service tiers
    - text options for verbosity, and output format
    - topK, topP, top_logprobs
    - truncation options
    - stateful params:
      - background
      - conversation
      - prompt
      - previous_response_id

- OutputParams

  - Supported: input/output tokens and cached tokens usage, error
  - Not supported:
    - All others are passed as opaque Details.

- Input output content (i.e messages and system prompt):

  - Input message: all except stateful properties within image and file content (file_id as of now).
  - Output message:

    - Support: Text input, URL citations i.e web search result.
    - Don't support: Logprobs and few citations
      - file, container, filepath. Need to distinguish which one is useful for actual stateless api operation, vs which is strictly for stateful flows.

  - Reasoning: all supported.

  - Function call, custom tool call: all supported.
  - Function call output, custom tool call output: All supported except stateful properties (file_id as of now)
  - Image generation call: Not supported yet.
  - Web search tool call and result: all supported.
  - Item reference: Not sure what it is used for. Most probably some stateful thing.

  - Not supported:

    - Compaction - something stateful
    - Tool Calls and Outputs for tools: Code interpreter, local shell, shell, apply patch
    - MCP list tools, approval req/resp, tool call.

- Tool choice:

  - [ ] Deferred

    - [ ] File search: it is about using files uplaoded to openai vector stores. may look at it when we do vector stores thing.
    - [ ] Remote MCP: decide when to do mcp integrations.
    - [ ] Remote Connectors: decide when to do mcp integrations. This is better as it would give concrete access to remote data sources.
    - [ ] Image generation.

  - [ ] Deferred/Think through

    - [ ] Shell tool: this is a fixed schema for a host driven shell tool implementation. Better to have a much more controlled and safe and tunable in app shell calls and execute tool and flow. arbitrary shell commands in loop can be quite dangerous overall.
    - [ ] Apply patch tool: this is a fixed schema for a host driven patch generation and apply tool. it allows models to generate patches (mostly will be some internal schema conformance thing) and we have to apply them locally.
    - [ ] The utility of these tools is mostly that you dont have to give input schema for this in function calling. This may be useful over host driven similar tools only if there is some cost saving associated with it.

  - [ ] Don't

    - [ ] Computer use: Clicks and image captures of screenshots from computer. Not sure about the utility of it yet.
    - [ ] Code interpreter: about executing python in openai's own sandbox. not sure about the utility of it yet.
    - [ ] Local shell tool: outdated, and available only on codex mini.

## Nitpicks

- Anthropic: Why is server tool use block with websearch input as any??
- Anthropic: Why have content block source in document source as input?
- [ ] Image output
  - [ ] anthropic doesnt support image generation
  - [ ] openai supports image output via the image generation tool, not as input output content message
  - [ ] google generate content has image gen as direct input output content but via dedicated models.
  - [ ] Given that what should be whee is not known, this should be deferred.
