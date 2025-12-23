package debugclient

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"runtime/debug"
	"strings"

	"github.com/ppipada/inference-go/internal/logutil"
	"github.com/ppipada/inference-go/spec"
)

// Ensure HTTPCompletionDebugger implements the public CompletionDebugger
// interface.
var _ spec.CompletionDebugger = (*HTTPCompletionDebugger)(nil)

// HTTPCompletionDebugger is a spec.CompletionDebugger backed by the HTTP
// instrumentation in this package. It uses DebugConfig to control what is
// captured and how it is scrubbed.
type HTTPCompletionDebugger struct {
	cfg DebugConfig
}

// NewHTTPCompletionDebugger constructs a CompletionDebugger that instruments
// HTTP traffic using LogTransport and exposes a scrubbed, opaque debug blob
// suitable for attachment to FetchCompletionResponse.DebugDetails.
func NewHTTPCompletionDebugger(cfg DebugConfig) spec.CompletionDebugger {
	// Make a shallow copy to avoid caller mutation.
	return &HTTPCompletionDebugger{cfg: cfg}
}

// WrapContext attaches a DebugHTTPResponse container to the context so that
// LogTransport can populate it.
func (d *HTTPCompletionDebugger) WrapContext(ctx context.Context) context.Context {
	if !d.cfg.Enabled {
		return ctx
	}
	return AddDebugResponseToCtx(ctx)
}

// HTTPClient returns an http.Client instrumented with LogTransport. If
// debugging is disabled, nil is returned and the provider SDK's default
// client is used instead.
func (d *HTTPCompletionDebugger) HTTPClient() *http.Client {
	if !d.cfg.Enabled {
		return nil
	}
	return NewDebugHTTPClient(d.cfg)
}

// BuildDebugDetails summarizes the captured HTTP debug information (if any),
// the final SDK response object and any error into a single opaque structure.
// The inference layer treats this as opaque and simply attaches it to
// FetchCompletionResponse.DebugDetails.
func (d *HTTPCompletionDebugger) BuildDebugDetails(
	ctx context.Context,
	fullResponse any,
	respErr error,
	isNilResp bool,
) any {
	defer func() {
		if r := recover(); r != nil {
			logutil.Error("debugclient.BuildDebugDetails panic",
				"panic", r,
				"stack", string(debug.Stack()))
		}
	}()

	if !d.cfg.Enabled {
		return nil
	}

	debugMap := map[string]any{
		"requestDetails":  map[string]any{},
		"responseDetails": map[string]any{},
		"errorDetails":    map[string]any{},
	}

	debugResp, _ := GetDebugHTTPResponse(ctx)

	// Always attach request/response debug info from the HTTP layer if available.
	if debugResp != nil {
		if debugResp.RequestDetails != nil {
			if m, err := structWithJSONTagsToMap(debugResp.RequestDetails); err == nil {
				debugMap["requestDetails"] = m
			}
		}
		if debugResp.ResponseDetails != nil {
			if m, err := structWithJSONTagsToMap(debugResp.ResponseDetails); err == nil {
				debugMap["responseDetails"] = m
			}
		}
	}

	// If we have a final SDK object, scrub and attach it as the response body.
	if fullResponse != nil {
		if m, err := structWithJSONTagsToMap(fullResponse); err == nil {
			if rd, ok := debugMap["responseDetails"].(map[string]any); ok {
				rd["data"] = ScrubAnyForDebug(m, true)
			} else {
				debugMap["responseDetails"] = map[string]any{
					"data": ScrubAnyForDebug(m, true),
				}
			}
		}
	}

	// Gather error-message fragments.
	var msgParts []string
	if debugResp != nil && debugResp.ErrorDetails != nil {
		if m := strings.TrimSpace(debugResp.ErrorDetails.Message); m != "" {
			msgParts = append(msgParts, m)
		}
	}
	if respErr != nil {
		msgParts = append(msgParts, respErr.Error())
	}
	if isNilResp {
		msgParts = append(msgParts, "got nil response from LLM api")
	}

	if len(msgParts) == 0 {
		return debugMap
	}

	if debugResp != nil && debugResp.ErrorDetails != nil {
		ed := *debugResp.ErrorDetails
		ed.Message = strings.Join(msgParts, "; ")
		if m, err := structWithJSONTagsToMap(ed); err == nil {
			debugMap["errorDetails"] = m
		}
	} else {
		debugMap["errorDetails"] = map[string]any{
			"message": strings.Join(msgParts, "; "),
		}
	}

	return debugMap
}

func structWithJSONTagsToMap(data any) (map[string]any, error) {
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
