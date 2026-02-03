# 🧠 **Chat Completions API — Request Body Reference (Markdown)**

**Endpoint:**
`POST https://api.openai.com/v1/chat/completions`
Creates a model response for the given chat conversation.

---

## **Request Body**

```jsonc
{
  "model": "string",
  "messages": [
    {
      "role": "string",
      "name": "string?",
      "content": "string | array"
    }
  ],
  "temperature": "number?",
  "top_p": "number?",
  "presence_penalty": "number?",
  "frequency_penalty": "number?",
  "max_tokens": "integer?",
  "stop": "string | array | null",
  "logit_bias": "{ [token_id:string]: number }?",
  "seed": "integer | null?",
  "prompt_cache_key": "string?",
  "prompt_cache_retention": "string?",
  "reasoning_effort": "string?",
  "response_format": {
    "type": "string",
    "json_schema": {
      "name": "string",
      "description": "string?",
      "schema": "object?",
      "strict": "boolean?"
    }
  }?,
  "safety_identifier": "string?",
  "service_tier": "string?"
}
```

---

## **Field Definitions**

---

### ✏️ **`model`** (string)

ID of the model used to generate the chat completion.

---

## 👥 **`messages`** (array) — _Required_

List of message objects representing the conversation history.

Each message object:

```markdown
{
"role": string, // e.g., "system", "user", "assistant", "developer"
"name": string?, // Optional participant name
"content": string | array // The actual message content
}
```

---

### 🗨️ **Message `content` Types**

- **Text**

  ```markdown
  string
  ```

- **Array of content parts**

  ```markdown
  [
  { "type": "text", "text": "string" },
  { "type": "image_url", "url": "string", "detail": "string?" },
  { "type": "input_audio", "data": "string", "format": "string" },
  { "type": "file", "file_data": "string?", "file_id": "string?", "filename": "string?" }
  ]
  ```

_Content part types supported differ by model._

---

## ⚙️ **Sampling & Generation Controls**

| Field               | Type     | Description                                 |
| ------------------- | -------- | ------------------------------------------- | --------------------------------------- | ------------------------------ |
| `temperature`       | number?  | Sampling temperature (controls randomness). |
| `top_p`             | number?  | Nucleus sampling parameter.                 |
| `presence_penalty`  | number?  | Penalizes new token topics.                 |
| `frequency_penalty` | number?  | Penalizes token frequency repetition.       |
| `max_tokens`        | integer? | Max tokens to generate.                     |
| `stop`              | string   | array                                       | null                                    | Stop sequences for generation. |
| `logit_bias`        | object?  | Token bias map.                             |
| `seed`              | integer  | null?                                       | (Beta) seed for deterministic sampling. |

All optional.

---

## 🧠 **Prompt & Model Caching**

| Field                    | Type    |
| ------------------------ | ------- |
| `prompt_cache_key`       | string? |
| `prompt_cache_retention` | string? |

Used to optimize repeated prefixes.

---

## 🧩 **Reasoning Effort**

**`reasoning_effort`** (string?)
Optional constraint for reasoning effort. Values include:

- `none`
- `minimal`
- `low`
- `medium`
- `high`
- `xhigh`

- _Behavior varies by model._

---

## 📦 **Structured Output / Response Format**

**`response_format`** (object?)

```markdown
{
"type": "text" | "json_schema" | "json_object",
"json_schema": {
"name": "string",
"description": "string?",
"schema": "object?",
"strict": "boolean?"
}
}
```

- `type`: Type of structured output desired.
- `json_schema`: JSON Schema config when using structured outputs.
- `strict`: If `true`, model enforces schema rules.

---

## 🛡️ **Safety & Policies**

| Field               | Type    | Notes                                       |
| ------------------- | ------- | ------------------------------------------- |
| `safety_identifier` | string? | Stable ID to help detect policy violations. |

Helps identify your application usage patterns.

---

## ⚡ **Service Tier**

**`service_tier`** (string?)
Controls processing tier (`auto`, `default`, `flex`, `priority`). Behavior may vary per model.

---

## ✅ **Summary — All Available Fields**

| Field                    | Type    | Required |
| ------------------------ | ------- | -------- | ---- | --- |
| `model`                  | string  | ✔️       |
| `messages`               | array   | ✔️       |
| `temperature`            | number  | ❌       |
| `top_p`                  | number  | ❌       |
| `presence_penalty`       | number  | ❌       |
| `frequency_penalty`      | number  | ❌       |
| `max_tokens`             | integer | ❌       |
| `stop`                   | string  | array    | null | ❌  |
| `logit_bias`             | object  | ❌       |
| `seed`                   | integer | null     | ❌   |
| `prompt_cache_key`       | string  | ❌       |
| `prompt_cache_retention` | string  | ❌       |
| `reasoning_effort`       | string  | ❌       |
| `response_format`        | object  | ❌       |
| `safety_identifier`      | string  | ❌       |
| `service_tier`           | string  | ❌       |
