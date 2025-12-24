package debugclient

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
)

func redactHeaders(headers map[string]any) map[string]any {
	if headers == nil {
		return nil
	}
	out := make(map[string]any, len(headers))
	for k, v := range headers {
		if containsSensitiveKey(k) {
			out[k] = maskToken
		} else {
			out[k] = v
		}
	}
	return out
}

// containsSensitiveKey checks if a key contains any sensitive keywords.
func containsSensitiveKey(key string) bool {
	lk := strings.ToLower(key)

	// Exact matches for common secret-bearing fields.
	if slices.Contains(sensitiveKeys, lk) {
		return true
	}

	// Heuristic: names ending with "_key" or "-key" are often API keys.
	if strings.HasSuffix(lk, "_key") || strings.HasSuffix(lk, "-key") {
		return true
	}

	return false
}

// looksLikeBase64 heuristically detects large base64 strings or data URLs.
func looksLikeBase64(s string) bool {
	if len(s) < 128 {
		return false
	}
	// Common pattern: data URLs.
	if strings.Contains(s, "base64,") {
		return true
	}

	// Heuristic: characters limited to base64 charset.
	for _, r := range s {
		switch {
		case r >= 'A' && r <= 'Z':
		case r >= 'a' && r <= 'z':
		case r >= '0' && r <= '9':
		case r == '+', r == '/', r == '=', r == '\n', r == '\r':
		default:
			return false
		}
	}

	// Optionally require length to be multiple of 4.
	if len(s)%4 != 0 {
		return false
	}
	return true
}

// pointerOf returns a stable pointer identity for maps and slices, or 0
// otherwise; used for cycle detection.
func pointerOf(x any) uintptr {
	rv := reflect.ValueOf(x)
	switch rv.Kind() {
	case reflect.Map, reflect.Slice:
		if rv.IsNil() {
			return 0
		}
		return rv.Pointer()
	default:
		return 0
	}
}

// shellQuote quotes a string for POSIX shells using single quotes.
func shellQuote(s string) string {
	// Pattern ' -> '\''.
	return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'"
}

// getDetailsStr pretty-prints any value as JSON for logging.
func getDetailsStr(v any) string {
	s, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf("Could not marshal object to JSON: %+v", v)
	}
	return string(s)
}
