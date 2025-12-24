package debugclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	"github.com/ppipada/inference-go/internal/logutil"
)

type contextKey string

const ctxKeyHTTPDebugState = contextKey("debugHTTPState")

// logTransport is a custom http.RoundTripper that captures HTTP requests and
// responses (including bodies) according to DebugConfig.
type logTransport struct {
	base http.RoundTripper
	cfg  DebugConfig
}

func (t *logTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}

	if t.cfg.Disable {
		// Debugging disabled; just pass through.
		return base.RoundTrip(req)
	}

	ctx := req.Context()
	state, _ := httpDebugStateFromContext(ctx)
	if state == nil {
		// Best-effort container (only visible to logs in this RoundTrip).
		state = &HTTPDebugState{}
	}

	// Capture request details (including optional body).
	reqDetails := captureRequestDetails(req, t.cfg)
	state.RequestDetails = reqDetails

	if t.cfg.LogToSlog {
		logutil.Debug("http_debug: request", "details", getDetailsStr(reqDetails))
	}

	// Perform the request.
	resp, err := base.RoundTrip(req)

	// Capture response details (headers, status, and possibly body).
	var respDetails *APIResponseDetails
	if resp != nil {
		respDetails = captureResponseDetails(resp, t.cfg, state)
		state.ResponseDetails = respDetails
	}

	// Capture error details if an error occurred.
	if err != nil {
		state.ErrorDetails = &APIErrorDetails{
			Message:  err.Error(),
			Request:  reqDetails,
			Response: respDetails,
		}
	}

	if t.cfg.LogToSlog {
		if respDetails != nil {
			logutil.Debug("http_debug: response", "details", getDetailsStr(respDetails))
		}
		if state.ErrorDetails != nil {
			logutil.Debug("http_debug: error", "details", getDetailsStr(state.ErrorDetails))
		}
	}

	return resp, err
}

func captureRequestDetails(req *http.Request, cfg DebugConfig) *APIRequestDetails {
	if req == nil {
		return nil
	}
	headers := make(map[string]any, len(req.Header))
	for key, values := range req.Header {
		headers[key] = strings.Join(values, ", ")
	}
	headers = redactHeaders(headers)

	params := make(map[string]any)
	if req.URL != nil {
		q := req.URL.Query()
		for key := range q {
			vals := q[key]
			if len(vals) == 1 {
				params[key] = vals[0]
			} else if len(vals) > 1 {
				params[key] = vals
			}
		}
	}

	var data any
	if !cfg.DisableRequestBody && req.Body != nil {
		bodyBytes, err := io.ReadAll(req.Body)
		if err == nil && len(bodyBytes) > 0 {
			data = sanitizeBodyForDebug(bodyBytes, true, cfg)
			// Reset body so it can be read by the underlying transport & SDK.
			req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		}
	} else if cfg.DisableRequestBody && req.Body != nil {
		// Indicate that body exists but was intentionally not captured.
		data = "[omitted: request body not captured by debug configuration]"
	}

	urlStr := ""
	if req.URL != nil {
		urlStr = req.URL.String()
	}
	method := req.Method

	apireq := &APIRequestDetails{
		URL:     &urlStr,
		Method:  &method,
		Headers: headers,
		Params:  params,
		Data:    data,
	}

	curl := generateCurlCommand(apireq)
	apireq.CurlCommand = &curl

	return apireq
}

func captureResponseDetails(
	resp *http.Response,
	cfg DebugConfig,
	state *HTTPDebugState,
) *APIResponseDetails {
	if resp == nil {
		return nil
	}

	headers := make(map[string]any, len(resp.Header))
	for key, values := range resp.Header {
		headers[key] = strings.Join(values, ", ")
	}
	headers = redactHeaders(headers)

	respDetails := &APIResponseDetails{
		Status:  resp.StatusCode,
		Headers: headers,
	}

	// Wrap the body if we want to capture it.
	if !cfg.DisableResponseBody && resp.Body != nil {
		buffer := new(bytes.Buffer)
		resp.Body = &loggingReadCloser{
			ReadCloser: resp.Body,
			buf:        buffer,
			state:      state,
			cfg:        cfg,
		}
	}

	return respDetails
}

// generateCurlCommand builds a (mostly) copy-pasteable curl command from
// apiRequestDetails. It uses the already-redacted Data and Headers.
func generateCurlCommand(config *APIRequestDetails) string {
	if config == nil || config.URL == nil || config.Method == nil {
		return ""
	}

	var b strings.Builder

	method := strings.ToUpper(*config.Method)
	b.WriteString("curl")
	if method != "" {
		b.WriteString(" -X ")
		b.WriteString(method)
	}

	if config.URL != nil {
		escapedURL := shellQuote(*config.URL)
		b.WriteString(" ")
		b.WriteString(escapedURL)
	}

	// Headers (sorted for stability).
	if len(config.Headers) > 0 {
		keys := make([]string, 0, len(config.Headers))
		for k := range config.Headers {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			v := config.Headers[k]
			headerStr := fmt.Sprintf("%s: %v", k, v)
			b.WriteString(" \\\n  -H ")
			b.WriteString(shellQuote(headerStr))
		}
	}

	if config.Data != nil {
		bodyBytes, err := json.MarshalIndent(config.Data, "", "  ")
		if err == nil {
			b.WriteString(" \\\n  --data-raw ")
			b.WriteString(shellQuote(string(bodyBytes)))
		}
	}

	return b.String()
}

// withHTTPDebugState sets up an HTTPDebugState container on the context.
// All SDK calls that should capture HTTP debug must use this context.
func withHTTPDebugState(ctx context.Context) context.Context {
	state := &HTTPDebugState{}
	return context.WithValue(ctx, ctxKeyHTTPDebugState, state)
}

// httpDebugStateFromContext retrieves the HTTPDebugState from context.
func httpDebugStateFromContext(ctx context.Context) (*HTTPDebugState, bool) {
	state, ok := ctx.Value(ctxKeyHTTPDebugState).(*HTTPDebugState)
	return state, ok
}

func structToMap(data any) (map[string]any, error) {
	if data == nil {
		return nil, errors.New("input data cannot be nil")
	}
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal struct to JSON: %w", err)
	}
	var result map[string]any
	if err := json.Unmarshal(jsonData, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON to map: %w", err)
	}
	return result, nil
}
