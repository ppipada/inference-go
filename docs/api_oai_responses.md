# **OpenAI Responses API — Comprehensive Parameter Reference (Markdown)**

> **Endpoint:** `POST https://api.openai.com/v1/responses`
> Creates a model response. Supports text, image, structured outputs, tools, reasoning, streaming, and more.

---

## 📌 **Request Body (JSON)**

```jsonc
{
  "model": "string",
  "input": string | array,
  "instructions": "string",
  "conversation": string | object,
  "previous_response_id": "string",
  "include": [ "string" ],
  "temperature": number,
  "top_p": number,
  "top_logprobs": number,
  "truncation": "string",
  "store": boolean,
  "stream": boolean,
  "stream_options": object,
  "background": boolean,
  "tools": [ object ],
  "tool_choice": string | object,
  "reasoning": object,
  "prompt": object,
  "prompt_cache_key": "string",
  "prompt_cache_retention": "string",
  "metadata": { [key: string]: string },
  "safety_identifier": "string",
  "service_tier": "string",
  "text": object
}
```

---

## 📌 **Field Reference (Markdown)**

---

### **1. `model`**

**Type:** `string`
Model identifier (e.g., `"gpt-5.2"`, `"gpt-4o"`) specifying which model should generate the response.

---

### **2. `input`**

**Type:** `string | array`
Input items the model should process. Can be:

- Plain text string
- Array of structured inputs (text, image, references)

Examples:

```jsonc
"input": "Translate this text"
```

or

```jsonc
"input": [
  {
    "role": "user",
    "content": [ { "type": "text", "text": "Hello!" } ]
  }
]
```

_Exact structured item types are defined by the Responses API spec._

---

### **3. `instructions`**

**Type:** `string`
System or developer instructions that guide model behavior.

---

### **4. `conversation`**

**Type:** `string | object`
Conversation ID or object to attach this response to an existing conversation.

---

### **5. `previous_response_id`**

**Type:** `string`
ID of a prior response to continue a thread. Mutually exclusive with `conversation`.

---

### **6. `include`**

**Type:** `array[string]`
Specifies additional output items to include. Example values include:

- `"web_search_call.action.sources"`
- `"file_search_call.results"`
- `"message.input_image.image_url"`
- `"message.output_text.logprobs"`

_This list is exactly as documented._

---

### **7. Generation Controls**

#### `temperature`

**Type:** `number` — controls randomness.

#### `top_p`

**Type:** `number` — nucleus sampling parameter.

#### `top_logprobs`

**Type:** `integer` — returns log-probabilities for tokens.

#### `truncation`

**Type:** `string` — context truncation strategy.

---

### **8. Response Behavior**

#### `store`

**Type:** `boolean` — whether the response should be stored.

#### `stream`

**Type:** `boolean` — whether the response is streamed as server-sent events.

---

### **9. `stream_options`**

**Type:** `object`
Stream configuration object. Known sub-fields _may include_:

```markdown
{
"sep": "string" // delimiter between stream events
}
```

_Public docs don’t fully list this._

---

### **10. `background`**

**Type:** `boolean`
If `true`, run the task in the background and return an ID immediately.

---

### **11. `tools`**

**Type:** `array[object]`
Array of tool definitions. Each tool object may contain:

```markdown
{
"type": "string", // tool type (e.g., web_search)
"name": "string", // tool name (optional)
"description": "string", // short description
"parameters": { ... } // tool-specific fields
}
```

_Because tool definitions vary by type, the exact schema depends on the tool._

---

### **12. `tool_choice`**

**Type:** `string | object`
Controls which tools the model may choose:

- `"auto"` — default behavior
- `"required"` — force a tool call
- Object specifying `{ "name": "...", "allowed_tools": [ ... ] }`

_Public docs hint at these options._

---

### **13. `reasoning`**

**Type:** `object`
Configuration for chain-of-thought reasoning. Known fields:

```markdown
{
"effort": "none | low | medium | high | xhigh",
"verbosity": "low | medium | high"
}
```

_These parameters exist in the broader guide pages._

---

### **14. Prompt Template Fields**

#### `prompt`

**Type:** `object`
Reference to a reusable prompt template:

```markdown
{
"id": "string",
"version": number,
"variables": { [key: string]: any }
}
```

_Used for variable prompt templates._

---

#### `prompt_cache_key`

**Type:** `string`
Cache key for prompt caching.

#### `prompt_cache_retention`

**Type:** `string`
How long to retain prompt cache (e.g., `"24h"`).

---

### **15. Custom Metadata**

#### `metadata`

**Type:** `{ [key: string]: string }`
User-defined key/value data attached to the response.

#### `safety_identifier`

**Type:** `string`
Identifier for safety/policy tracking.

---

### **16. Service Tier**

#### `service_tier`

**Type:** `string`
Quality/performance tier (e.g., `"auto"`, `"default"`, `"priority"`, `"flex"`).

---

### **17. `text` Block (Text Output Config)**

**Type:** `object`
Configures how text output is generated. May include:

```markdown
{
"verbosity": "low|medium|high",
"format": {
"type": "text | json_schema | json_object",
"json_schema": { ... },
"strict": boolean
}
}
```

- **`verbosity`** – controls text detail
- **`format.type`** – `text`, `json_schema`, or `json_object`
- **`json_schema`** – user-provided JSON Schema definition
- **`strict`** – enforce strict schema adherence

---

## 📌 **Complete Field Summary Table**

| Field                    | Type          |
| ------------------------ | ------------- | ------ |
| `model`                  | string        |
| `input`                  | string        | array  |
| `instructions`           | string        |
| `conversation`           | string        | object |
| `previous_response_id`   | string        |
| `include`                | array[string] |
| `temperature`            | number        |
| `top_p`                  | number        |
| `top_logprobs`           | number        |
| `truncation`             | string        |
| `store`                  | boolean       |
| `stream`                 | boolean       |
| `stream_options`         | object        |
| `background`             | boolean       |
| `tools`                  | array[object] |
| `tool_choice`            | string        | object |
| `reasoning`              | object        |
| `prompt`                 | object        |
| `prompt_cache_key`       | string        |
| `prompt_cache_retention` | string        |
| `metadata`               | map           |
| `safety_identifier`      | string        |
| `service_tier`           | string        |
| `text`                   | object        |

---
