package inference

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	"github.com/ppipada/inference-go/internal/anthropicsdk"
	"github.com/ppipada/inference-go/internal/debugclient"
	"github.com/ppipada/inference-go/internal/logutil"
	"github.com/ppipada/inference-go/internal/openaichatsdk"
	"github.com/ppipada/inference-go/internal/openairesponsessdk"
	"github.com/ppipada/inference-go/internal/sdkutil"
	"github.com/ppipada/inference-go/spec"
)

// LoggerBuilder returns the slog.Logger to be used by this ProviderSet. The
// builder is evaluated during NewProviderSetAPI; passing nil or returning nil
// results in a no-op logger being installed via logutil.SetLogger.
type LoggerBuilder func() *slog.Logger

// DebugClientBuilder constructs a CompletionDebugger for a given provider. A
// nil builder or a nil returned debugger disable debugging for that provider.
type DebugClientBuilder func(p spec.ProviderParam) spec.CompletionDebugger

type ProviderSetAPI struct {
	mu sync.RWMutex

	providers          map[spec.ProviderName]spec.CompletionProvider
	loggerBuilder      LoggerBuilder
	debugClientBuilder DebugClientBuilder
}

// ProviderSetOption configures optional behavior for ProviderSetAPI.
type ProviderSetOption func(*ProviderSetAPI)

// WithLoggerBuilder installs a process-wide logger for this SDK. The builder
// is evaluated during NewProviderSetAPI; passing nil results in a no-op logger.
func WithLoggerBuilder(builder LoggerBuilder) ProviderSetOption {
	return func(ps *ProviderSetAPI) {
		ps.loggerBuilder = builder
	}
}

// WithDebugClientBuilder configures a CompletionDebugger factory. The builder
// is invoked once per provider when it is added. Returning nil disables
// debugging for that provider.
func WithDebugClientBuilder(builder DebugClientBuilder) ProviderSetOption {
	return func(ps *ProviderSetAPI) {
		ps.debugClientBuilder = builder
	}
}

// DebugOptions provides a high-level way to configure the built-in HTTP
// debugger based on internal/debugclient. This is a convenience on top of
// WithDebugClientBuilder; callers that need full control can provide their
// own builder instead.
type DebugOptions struct {
	Enabled             bool
	CaptureRequestBody  bool
	CaptureResponseBody bool
	StripContent        bool
	LogToLogger         bool
}

// WithHTTPDebugOptions installs a DebugClientBuilder that uses the internal
// HTTP debugger with the provided options.
func WithHTTPDebugOptions(opts DebugOptions) ProviderSetOption {
	return func(ps *ProviderSetAPI) {
		if !opts.Enabled {
			ps.debugClientBuilder = nil
			return
		}
		cfg := debugclient.DefaultDebugConfig
		cfg.Enabled = true
		if opts.CaptureRequestBody {
			cfg.CaptureRequestBody = true
		}
		if opts.CaptureResponseBody {
			cfg.CaptureResponseBody = true
		}
		if opts.StripContent {
			cfg.StripContent = true
		}
		if opts.LogToLogger {
			cfg.LogToSlog = true
		}

		ps.debugClientBuilder = func(p spec.ProviderParam) spec.CompletionDebugger {
			return debugclient.NewHTTPCompletionDebugger(cfg)
		}
	}
}

// NewProviderSetAPI creates a new ProviderSet and installs the process-wide
// logger used by this SDK. The logger is chosen via WithLoggerBuilder; if no
// builder is provided or it returns nil, a no-op logger is used.
func NewProviderSetAPI(
	opts ...ProviderSetOption,
) (*ProviderSetAPI, error) {
	ps := &ProviderSetAPI{
		providers: map[spec.ProviderName]spec.CompletionProvider{},
	}

	for _, opt := range opts {
		if opt != nil {
			opt(ps)
		}
	}

	if ps.loggerBuilder != nil {
		logutil.SetDefault(ps.loggerBuilder())
	} else {
		logutil.SetDefault(nil)
	}

	return ps, nil
}

type AddProviderConfig struct {
	SDKType                  spec.ProviderSDKType `json:"sdkType"`
	Origin                   string               `json:"origin"`
	ChatCompletionPathPrefix string               `json:"chatCompletionPathPrefix"`
	APIKeyHeaderKey          string               `json:"apiKeyHeaderKey"`
	DefaultHeaders           map[string]string    `json:"defaultHeaders"`
}

func (ps *ProviderSetAPI) AddProvider(
	ctx context.Context,
	provider spec.ProviderName,
	config *AddProviderConfig,
) (*spec.ProviderParam, error) {
	if config == nil || provider == "" || config.Origin == "" {
		return nil, errors.New("invalid params")
	}

	ps.mu.Lock()
	defer ps.mu.Unlock()

	_, exists := ps.providers[provider]
	if exists {
		return nil, errors.New(
			"invalid provider: cannot add a provider with same name as an existing provider, delete first",
		)
	}
	if ok := isProviderSDKTypeSupported(config.SDKType); !ok {
		return nil, errors.New("unsupported provider api type")
	}

	providerInfo := spec.ProviderParam{
		Name:                     provider,
		SDKType:                  config.SDKType,
		APIKey:                   "",
		Origin:                   config.Origin,
		ChatCompletionPathPrefix: config.ChatCompletionPathPrefix,
		APIKeyHeaderKey:          config.APIKeyHeaderKey,
		DefaultHeaders:           config.DefaultHeaders,
	}

	var dbg spec.CompletionDebugger
	if ps.debugClientBuilder != nil {
		dbg = ps.debugClientBuilder(providerInfo)
	}

	cp, err := getProviderAPI(providerInfo, dbg)
	if err != nil {
		return nil, err
	}
	ps.providers[provider] = cp

	logutil.Info("add provider", "name", provider)

	return cp.GetProviderInfo(ctx), nil
}

func (ps *ProviderSetAPI) DeleteProvider(
	ctx context.Context,
	provider spec.ProviderName,
) error {
	if provider == "" {
		return errors.New("got empty provider input")
	}
	ps.mu.Lock()
	p, exists := ps.providers[provider]
	if !exists {
		ps.mu.Unlock()
		return errors.New("invalid provider: provider does not exist")
	}
	delete(ps.providers, provider)
	ps.mu.Unlock()

	// Best-effort cleanup outside the lock.
	_ = p.DeInitLLM(ctx)
	logutil.Info("deleteProvider", "name", provider)

	return nil
}

type SetProviderAPIKeyRequestBody struct {
	APIKey string `json:"apiKey" required:"true"`
}

type SetProviderAPIKeyResponse struct{}

// SetProviderAPIKey sets the key for a given provider.
func (ps *ProviderSetAPI) SetProviderAPIKey(
	ctx context.Context,
	provider spec.ProviderName,
	apiKey string,
) error {
	ps.mu.RLock()
	p, exists := ps.providers[provider]
	ps.mu.RUnlock()
	if !exists {
		return errors.New("invalid provider")
	}

	if apiKey == "" {
		// Clear the stored key as well as de-initialize the client.
		if info := p.GetProviderInfo(ctx); info != nil {
			info.APIKey = ""
		}
		return p.DeInitLLM(ctx)

	}
	err := p.SetProviderAPIKey(
		ctx,
		apiKey,
	)
	if err != nil {
		return err
	}
	err = p.InitLLM(ctx)
	if err != nil {
		return err
	}
	return nil
}

// FetchCompletion processes a completion request for a given provider.
func (ps *ProviderSetAPI) FetchCompletion(
	ctx context.Context,
	provider spec.ProviderName,
	fetchCompletionRequest *spec.FetchCompletionRequest,
	opts *spec.FetchCompletionOptions,
) (*spec.FetchCompletionResponse, error) {
	if provider == "" || fetchCompletionRequest == nil || len(fetchCompletionRequest.Inputs) == 0 ||
		fetchCompletionRequest.ModelParam.Name == "" {
		return nil, errors.New("got empty fetch completion input")
	}

	ps.mu.RLock()
	p, exists := ps.providers[provider]
	ps.mu.RUnlock()

	if !exists {
		return nil, errors.New("invalid provider")
	}

	reqCopy := *fetchCompletionRequest

	// If a max prompt length (in tokens) is configured, apply heuristic filtering.
	if reqCopy.ModelParam.MaxPromptLength > 0 {
		reqCopy.Inputs = sdkutil.FilterMessagesByTokenCount(
			fetchCompletionRequest.Inputs,
			reqCopy.ModelParam.MaxPromptLength,
		)
	}

	resp, err := p.FetchCompletion(
		ctx,
		&reqCopy,
		opts,
	)
	if err != nil {
		// Return any partial response we got alongside a contextual error.
		return resp, fmt.Errorf("fetch completion failed for provider %s: %w", provider, err)
	}

	return resp, nil
}

func isProviderSDKTypeSupported(t spec.ProviderSDKType) bool {
	if t == spec.ProviderSDKTypeAnthropic ||
		t == spec.ProviderSDKTypeOpenAIChatCompletions ||
		t == spec.ProviderSDKTypeOpenAIResponses {
		return true
	}
	return false
}

func getProviderAPI(p spec.ProviderParam, dbg spec.CompletionDebugger) (spec.CompletionProvider, error) {
	switch p.SDKType {
	case spec.ProviderSDKTypeAnthropic:
		return anthropicsdk.NewAnthropicMessagesAPI(p, dbg)

	case spec.ProviderSDKTypeOpenAIChatCompletions:
		return openaichatsdk.NewOpenAIChatCompletionsAPI(p, dbg)

	case spec.ProviderSDKTypeOpenAIResponses:
		return openairesponsessdk.NewOpenAIResponsesAPI(p, dbg)
	}

	return nil, errors.New("invalid provider api type")
}
