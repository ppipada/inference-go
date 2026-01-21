// Package inference provides a single, normalized interface for getting
// language model completions from multiple providers.
//
// The main entry point is ProviderSetAPI, which lets you:
//
//   - register one or more providers (Anthropic, OpenAI Chat Completions,
//     OpenAI Responses, ...),
//   - configure and rotate API keys,
//   - send normalized completion requests and receive normalized outputs,
//   - optionally stream partial text / reasoning and capture HTTP‑level
//     debug information.
package inference

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/flexigpt/inference-go/internal/anthropicsdk"

	"github.com/flexigpt/inference-go/internal/logutil"
	"github.com/flexigpt/inference-go/internal/openaichatsdk"
	"github.com/flexigpt/inference-go/internal/openairesponsessdk"
	"github.com/flexigpt/inference-go/internal/sdkutil"
	"github.com/flexigpt/inference-go/spec"
)

// DebugClientBuilder constructs a CompletionDebugger for a given provider. A
// nil builder or a nil returned debugger disable debugging for that provider.
type DebugClientBuilder func(p spec.ProviderParam) spec.CompletionDebugger

type ProviderSetAPI struct {
	mu sync.RWMutex

	providers          map[spec.ProviderName]spec.CompletionProvider
	logger             *slog.Logger
	debugClientBuilder DebugClientBuilder
}

// ProviderSetOption configures optional behavior for ProviderSetAPI.
type ProviderSetOption func(*ProviderSetAPI)

func WithLogger(logger *slog.Logger) ProviderSetOption {
	return func(ps *ProviderSetAPI) {
		ps.logger = logger
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

	if ps.logger != nil {
		logutil.SetDefault(ps.logger)
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
) (spec.ProviderParam, error) {
	if config == nil || provider == "" || config.Origin == "" {
		return spec.ProviderParam{}, errors.New("invalid params")
	}

	ps.mu.Lock()
	defer ps.mu.Unlock()

	_, exists := ps.providers[provider]
	if exists {
		return spec.ProviderParam{}, errors.New(
			"invalid provider: cannot add a provider with same name as an existing provider, delete first",
		)
	}
	if ok := isProviderSDKTypeSupported(config.SDKType); !ok {
		return spec.ProviderParam{}, errors.New("unsupported provider api type")
	}

	providerInfo := spec.ProviderParam{
		Name:                     provider,
		SDKType:                  config.SDKType,
		APIKey:                   "",
		Origin:                   config.Origin,
		ChatCompletionPathPrefix: config.ChatCompletionPathPrefix,
		APIKeyHeaderKey:          config.APIKeyHeaderKey,
		DefaultHeaders:           sdkutil.CloneStringMap(config.DefaultHeaders),
	}

	var dbg spec.CompletionDebugger
	if ps.debugClientBuilder != nil {
		dbg = ps.debugClientBuilder(providerInfo)
	}

	cp, err := getProviderAPI(providerInfo, dbg)
	if err != nil {
		return spec.ProviderParam{}, err
	}
	ps.providers[provider] = cp

	logutil.Info("add provider", "name", provider)

	return *cp.GetProviderInfo(ctx), nil
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

	apiKey = strings.TrimSpace(apiKey)
	err := p.SetProviderAPIKey(ctx, apiKey)
	if err != nil {
		return err
	}
	if apiKey == "" {
		return p.DeInitLLM(ctx)
	}
	return p.InitLLM(ctx)
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
