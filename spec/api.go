package spec

import (
	"context"
	"net/http"
)

// StreamContentKind enumerates the kinds of streaming events that can be delivered while a completion is in progress.
type StreamContentKind string

const (
	StreamContentKindText     StreamContentKind = "text"
	StreamContentKindThinking StreamContentKind = "thinking"
)

type StreamTextChunk struct {
	Text string `json:"text"`
}

type StreamThinkingChunk struct {
	Text string `json:"text"`
}

type StreamEvent struct {
	Kind StreamContentKind `json:"kind"`

	// Optional metadata to help consumers correlate events across models/providers.
	Provider ProviderName `json:"provider,omitempty"`
	Model    ModelName    `json:"model,omitempty"`

	// Exactly one of the below will be non-nil depending on Kind.
	Text     *StreamTextChunk     `json:"text,omitempty"`
	Thinking *StreamThinkingChunk `json:"thinking,omitempty"`
}

// StreamConfig controls low-level behavior of streaming delivery. All fields are optional; zero values mean "use
// library defaults".
type StreamConfig struct {
	// FlushIntervalMillis is the maximum delay between flushes of buffered stream data to the StreamHandler.
	FlushIntervalMillis int `json:"flushIntervalMillis,omitempty"`
	// FlushChunkSize is the approximate target size (in bytes/characters) for chunks passed to the StreamHandler.
	FlushChunkSize int `json:"flushChunkSize,omitempty"`
}

type StreamHandler func(event StreamEvent) error

// FetchCompletionOptions controls optional behaviors for FetchCompletion.
// A nil pointer is treated the same as &FetchCompletionOptions{}.
type FetchCompletionOptions struct {
	// StreamHandler, if non-nil, is invoked with incremental streaming events
	// when ModelParam.Stream is true. Returning a non-nil error will stop
	// streaming early and propagate that error back to the caller.
	StreamHandler StreamHandler `json:"-"`
	StreamConfig  *StreamConfig `json:"streamConfig,omitempty"`
}

// CompletionDebugger abstracts debugging/observability concerns for a single
// provider. Implementations may collect HTTP-level data, raw model responses,
// tracing information, etc. The inference layer treats the DebugDetails as
// opaque and never inspects its shape.
type CompletionDebugger interface {
	// WrapContext is called at the beginning of FetchCompletion. It may attach
	// any request-scoped state needed for later debug collection.
	WrapContext(ctx context.Context) context.Context

	// HTTPClient returns an HTTP client instrumented for debugging. If nil is
	// returned, the provider SDK's default client will be used.
	HTTPClient() *http.Client

	// BuildDebugDetails is called once, after the upstream SDK call completes (successfully or with error).
	// FullResponse is the raw SDK response object if available, otherwise nil.
	// IsNilResp indicates whether fullResponse was considered empty/invalid by the provider.
	BuildDebugDetails(ctx context.Context, fullResponse any, err error, isNilResp bool) any
}

type FetchCompletionResponse struct {
	Outputs      []OutputUnion `json:"outputs,omitempty"`
	Usage        *Usage        `json:"usage,omitempty"`
	Error        *Error        `json:"error,omitempty"`
	DebugDetails any           `json:"debugDetails,omitempty"`
}

type FetchCompletionRequest struct {
	ModelParam  ModelParam   `json:"modelParam"`
	Inputs      []InputUnion `json:"inputs"`
	ToolChoices []ToolChoice `json:"toolChoices,omitempty"`
}

type CompletionProvider interface {
	InitLLM(ctx context.Context) error
	DeInitLLM(ctx context.Context) error
	GetProviderInfo(ctx context.Context) *ProviderParam
	IsConfigured(ctx context.Context) bool
	SetProviderAPIKey(ctx context.Context, apiKey string) error
	FetchCompletion(
		ctx context.Context,
		fetchCompletionRequest *FetchCompletionRequest,
		opts *FetchCompletionOptions,
	) (*FetchCompletionResponse, error)
}
