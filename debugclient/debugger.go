package debugclient

import (
	"context"
	"net/http"
	"strings"

	"github.com/ppipada/inference-go/internal/sdkutil"
	"github.com/ppipada/inference-go/spec"
)

// DebugConfig controls how HTTP debug information is captured and redacted.
//
// The zero value corresponds to the default behavior:
//
//   - debugging enabled
//   - request/response bodies captured
//   - content (LLM text, large/base64 blobs) stripped/scrubbed
//   - no slog logging
type DebugConfig struct {
	// Disable turns off all debugging when true.
	Disable bool `json:"disable,omitempty"`

	// DisableRequestBody prevents capturing request bodies when true.
	DisableRequestBody bool `json:"disableRequestBody,omitempty"`

	// DisableResponseBody prevents capturing response bodies when true.
	DisableResponseBody bool `json:"disableResponseBody,omitempty"`

	// DisableContentStripping, if true, leaves LLM text content and large/base64
	// payloads untouched. When false (default), scrubbers remove user/assistant
	// text and large/base64 blobs while preserving other metadata.
	DisableContentStripping bool `json:"disableContentStripping,omitempty"`

	// LogToSlog logs HTTP request/response details at debug level when true.
	LogToSlog bool `json:"logToSlog,omitempty"`
}

// HTTPCompletionDebugger implements spec.CompletionDebugger using the HTTP
// instrumentation in this package. It produces an opaque debug payload
// suitable for attachment to FetchCompletionResponse.DebugDetails.
type HTTPCompletionDebugger struct {
	cfg DebugConfig
}

// NewHTTPCompletionDebugger constructs a CompletionDebugger that instruments
// HTTP traffic using an internal RoundTripper and produces a scrubbed debug blob from HTTP-level data and the raw
// provider response.
// Config may be nil; in that case DebugConfig{} (defaults) is used.
func NewHTTPCompletionDebugger(cfg *DebugConfig) spec.CompletionDebugger {
	var c DebugConfig
	if cfg != nil {
		c = *cfg
	}
	return &HTTPCompletionDebugger{cfg: c}
}

// HTTPClient implements spec.CompletionDebugger.HTTPClient.
func (d *HTTPCompletionDebugger) HTTPClient(base *http.Client) *http.Client {
	if d.cfg.Disable {
		return base
	}

	if base == nil {
		base = &http.Client{Transport: http.DefaultTransport}
	}
	rt := base.Transport
	if rt == nil {
		rt = http.DefaultTransport
	}

	clone := *base
	clone.Transport = &logTransport{
		base: rt,
		cfg:  d.cfg,
	}
	return &clone
}

type httpSpan struct {
	cfg  DebugConfig
	ctx  context.Context
	info *spec.CompletionSpanStart
}

// StartSpan implements spec.CompletionDebugger.StartSpan.
func (d *HTTPCompletionDebugger) StartSpan(
	ctx context.Context,
	info *spec.CompletionSpanStart,
) (context.Context, spec.CompletionSpan) {
	if d.cfg.Disable {
		return ctx, nil
	}

	ctx = withHTTPDebugState(ctx)
	span := &httpSpan{
		cfg:  d.cfg,
		ctx:  ctx,
		info: info,
	}
	return ctx, span
}

type APIRequestDetails struct {
	URL         *string        `json:"url,omitempty"`
	Method      *string        `json:"method,omitempty"`
	Headers     map[string]any `json:"headers,omitempty"`
	Params      map[string]any `json:"params,omitempty"`
	Data        any            `json:"data,omitempty"`
	Timeout     *int           `json:"timeout,omitempty"`
	CurlCommand *string        `json:"curlCommand,omitempty"`
}

type APIResponseDetails struct {
	Data           any                `json:"data,omitempty"`
	Status         int                `json:"status"`
	Headers        map[string]any     `json:"headers"`
	RequestDetails *APIRequestDetails `json:"requestDetails,omitempty"`
}

type APIErrorDetails struct {
	Message  string              `json:"message"`
	Request  *APIRequestDetails  `json:"requestDetails,omitempty"`
	Response *APIResponseDetails `json:"responseDetails,omitempty"`
}

// HTTPDebugState wraps HTTP debug info stored on the context.
type HTTPDebugState struct {
	RequestDetails  *APIRequestDetails  `json:"requestDetails"`
	ResponseDetails *APIResponseDetails `json:"responseDetails"`
	ErrorDetails    *APIErrorDetails    `json:"errorDetails"`
}

// End implements spec.CompletionSpan.End.
func (s *httpSpan) End(end *spec.CompletionSpanEnd) any {
	defer sdkutil.Recover("debugclient.httpSpan.End panic")

	if s.cfg.Disable {
		return nil
	}

	state, _ := httpDebugStateFromContext(s.ctx)

	debugMap := map[string]any{
		"requestDetails":  map[string]any{},
		"responseDetails": map[string]any{},
		"errorDetails":    map[string]any{},
	}

	// HTTP request/response from transport, if available.
	if state != nil {
		if state.RequestDetails != nil {
			if m, err := structToMap(state.RequestDetails); err == nil {
				debugMap["requestDetails"] = m
			}
		}
		if state.ResponseDetails != nil {
			if m, err := structToMap(state.ResponseDetails); err == nil {
				debugMap["responseDetails"] = m
			}
		}
	}

	// Raw provider response (e.g. *responses.Response), scrubbed and attached
	// under responseDetails.data.
	if end.ProviderResponse != nil {
		if m, err := structToMap(end.ProviderResponse); err == nil {
			strip := !s.cfg.DisableContentStripping
			scrubbed := scrubAnyForDebug(m, strip)

			if rd, ok := debugMap["responseDetails"].(map[string]any); ok {
				rd["data"] = scrubbed
			} else {
				debugMap["responseDetails"] = map[string]any{
					"data": scrubbed,
				}
			}
		}
	}

	// Compose error message from HTTP-level error + provider error.
	var msgParts []string
	if state != nil && state.ErrorDetails != nil {
		if m := strings.TrimSpace(state.ErrorDetails.Message); m != "" {
			msgParts = append(msgParts, m)
		}
	}
	if end.Err != nil {
		msgParts = append(msgParts, end.Err.Error())
	}

	if len(msgParts) == 0 {
		return debugMap
	}

	if state != nil && state.ErrorDetails != nil {
		ed := *state.ErrorDetails
		ed.Message = strings.Join(msgParts, "; ")
		if m, err := structToMap(ed); err == nil {
			debugMap["errorDetails"] = m
		}
	} else {
		debugMap["errorDetails"] = map[string]any{
			"message": strings.Join(msgParts, "; "),
		}
	}

	return debugMap
}
