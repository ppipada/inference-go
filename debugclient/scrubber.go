package debugclient

import (
	"encoding/json"
	"fmt"
	"strings"
)

const (
	maxScrubDepth              = 4096
	maskToken                  = "***"
	cycleToken                 = "<cycle>"
	depthToken                 = "<max-depth>"
	textStr                    = "text"
	contentStr                 = "content"
	deltaStr                   = "delta"
	ommitedTextContentStr      = "[omitted: llm text content]"
	ommitedEncryptedContentStr = "[omitted: encrypted content]"
)

// Sensitive keys to filter in headers and bodies.
var sensitiveKeys = []string{
	"authorization",
	"proxy-authorization",
	"api-key",
	"apikey",
	"api_key",
	"x-api-key",
}

type scrubber struct {
	cfg       DebugConfig
	isRequest bool
	seen      map[uintptr]struct{}
}

type scrubContext struct {
	insideMessage bool
	parentKey     string
}

// sanitizeBodyForDebug parses and redacts a JSON or text body according to
// DebugConfig. It returns the sanitized representation as 'any' suitable for
// apiRequestDetails.Data or apiResponseDetails.Data.
func sanitizeBodyForDebug(raw []byte, isRequest bool, cfg DebugConfig) any {
	if len(raw) == 0 {
		return nil
	}

	// Try to parse as JSON (objects or arrays).
	var decoded any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		// Not JSON; treat as plain text.
		s := string(raw)
		if !cfg.DisableContentStripping {
			return scrubPlainText(s)
		}
		return s
	}

	s := newScrubber(cfg, isRequest)
	return s.scrub(decoded, 0, scrubContext{})
}

// scrubPlainText applies minimal redaction to a non-JSON body.
func scrubPlainText(s string) any {
	if looksLikeBase64(s) {
		return fmt.Sprintf("[omitted: %d bytes base64 data]", len(s))
	}
	// For non-JSON plain text we don't have structure; just return as-is.
	return s
}

// scrubAnyForDebug applies the same sanitization logic as sanitizeBodyForDebug
// to an already-decoded value.
//
// StripContent governs whether LLM text/base64 content is stripped.
func scrubAnyForDebug(v any, stripContent bool) any {
	cfg := DebugConfig{DisableContentStripping: !stripContent}
	s := newScrubber(cfg, false)
	return s.scrub(v, 0, scrubContext{})
}

func newScrubber(cfg DebugConfig, isRequest bool) *scrubber {
	return &scrubber{
		cfg:       cfg,
		isRequest: isRequest,
		seen:      make(map[uintptr]struct{}),
	}
}

func (s *scrubber) scrubMap(m map[string]any, depth int, ctx scrubContext) any {
	if p := pointerOf(m); p != 0 {
		if _, ok := s.seen[p]; ok {
			return cycleToken
		}
		s.seen[p] = struct{}{}
		defer delete(s.seen, p)
	}

	// Detect chat "message" objects by role.
	insideMessage := ctx.insideMessage
	if roleRaw, ok := m["role"].(string); ok {
		role := strings.ToLower(strings.TrimSpace(roleRaw))
		if role == "user" || role == "assistant" {
			insideMessage = true
		}
	}

	out := make(map[string]any, len(m))
	for k, val := range m {
		lk := strings.ToLower(k)

		// Redact sensitive keys (API keys, Authorization, etc.).
		if containsSensitiveKey(lk) {
			out[k] = maskToken
			continue
		}

		childCtx := scrubContext{
			insideMessage: insideMessage,
			parentKey:     k,
		}

		// Strip message "content" for user/assistant messages.
		if !s.cfg.DisableContentStripping && insideMessage && lk == contentStr {
			out[k] = s.scrubMessageContent(val, depth+1, childCtx)
			continue
		}

		// Strip top-level LLM text fields like "input", "prompt", "query".
		if !s.cfg.DisableContentStripping && (lk == "input" || lk == "prompt" || lk == "query") {
			out[k] = s.scrubTopLevelText(val, depth+1, childCtx)
			continue
		}

		out[k] = s.scrub(val, depth+1, childCtx)
	}
	return out
}

func (s *scrubber) scrubSlice(arr []any, depth int, ctx scrubContext) any {
	if p := pointerOf(arr); p != 0 {
		if _, ok := s.seen[p]; ok {
			return cycleToken
		}
		s.seen[p] = struct{}{}
		defer delete(s.seen, p)
	}

	out := make([]any, len(arr))
	for i, elem := range arr {
		out[i] = s.scrub(elem, depth+1, ctx)
	}
	return out
}

func (s *scrubber) scrubString(str string, ctx scrubContext) any {
	// First, strip large/base64-like data.
	if !s.cfg.DisableContentStripping && looksLikeBase64(str) {
		return fmt.Sprintf("[omitted: %d bytes base64 data]", len(str))
	}

	// If we are inside a message, and this is likely a text field, scrub it.
	if !s.cfg.DisableContentStripping && ctx.insideMessage {
		lk := strings.ToLower(ctx.parentKey)
		if lk == textStr || lk == contentStr || lk == deltaStr {
			return ommitedTextContentStr
		} else if strings.Contains(lk, "encrypted") {
			return ommitedEncryptedContentStr
		}
	}

	return str
}

// scrubMessageContent handles the "content" field of a user/assistant message.
func (s *scrubber) scrubMessageContent(val any, depth int, ctx scrubContext) any {
	const placeholder = ommitedTextContentStr

	if s.cfg.DisableContentStripping {
		return s.scrub(val, depth, ctx)
	}

	switch vv := val.(type) {
	case string:
		return placeholder

	case []any:
		// Handle structured content segments, e.g. OpenAI input_text/output_text,
		// images, files, etc. We try to keep non-text segments while scrubbing
		// textual parts.
		out := make([]any, 0, len(vv))
		for _, seg := range vv {
			if segMap, ok := seg.(map[string]any); ok {
				out = append(out, s.scrubContentSegment(segMap, depth+1))
			} else {
				out = append(out, s.scrub(seg, depth+1, ctx))
			}
		}
		return out

	default:
		// Unknown structure; just replace with placeholder.
		return placeholder
	}
}

func (s *scrubber) scrubTopLevelText(val any, depth int, ctx scrubContext) any {
	// Treat this subtree as "inside message" for scrubbing purposes.
	ctx.insideMessage = true
	return s.scrub(val, depth, ctx)
}

// scrubContentSegment scrubs a single structured content segment of a message.
// Text segments have their text/content removed; other types keep metadata.
func (s *scrubber) scrubContentSegment(seg map[string]any, depth int) any {
	segTypeRaw, _ := seg["type"].(string)
	segType := strings.ToLower(strings.TrimSpace(segTypeRaw))

	out := make(map[string]any, len(seg))
	for k, v := range seg {
		lk := strings.ToLower(k)

		if containsSensitiveKey(lk) {
			out[k] = maskToken
			continue
		}

		// Textual segments: drop text/content.
		if !s.cfg.DisableContentStripping && (segType == "input_text" || segType == "output_text" ||
			segType == textStr || segType == "message") {
			if lk == textStr || lk == contentStr {
				out[k] = ommitedTextContentStr
				continue
			} else if strings.Contains(lk, "encrypted") {
				out[k] = ommitedEncryptedContentStr
				continue
			}
		}

		// For everything else, recurse normally. Base64 / binary values will be
		// stripped by scrubString.
		out[k] = s.scrub(v, depth+1, scrubContext{
			insideMessage: true,
			parentKey:     k,
		})
	}

	return out
}

func (s *scrubber) scrub(v any, depth int, ctx scrubContext) any {
	if depth > maxScrubDepth {
		return depthToken
	}

	switch vv := v.(type) {
	case map[string]any:
		return s.scrubMap(vv, depth, ctx)
	case []any:
		return s.scrubSlice(vv, depth, ctx)
	case string:
		return s.scrubString(vv, ctx)
	default:
		return vv
	}
}
