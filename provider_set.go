package inference

import (
	"context"
	"errors"
	"log/slog"
	"sync"

	"github.com/ppipada/inference-go/internal/anthropicsdk"
	"github.com/ppipada/inference-go/internal/openaichatsdk"
	"github.com/ppipada/inference-go/internal/openairesponsessdk"
	"github.com/ppipada/inference-go/internal/sdkutil"
	"github.com/ppipada/inference-go/spec"
)

type ProviderSetAPI struct {
	mu sync.RWMutex

	providers map[spec.ProviderName]spec.CompletionProvider
	debug     bool
}

// NewProviderSetAPI creates a new ProviderSet with the specified default provider.
func NewProviderSetAPI(
	debug bool,
) (*ProviderSetAPI, error) {
	return &ProviderSetAPI{
		providers: map[spec.ProviderName]spec.CompletionProvider{},
		debug:     debug,
	}, nil
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

	cp, err := getProviderAPI(providerInfo, ps.debug)
	if err != nil {
		return nil, err
	}
	ps.providers[provider] = cp

	slog.Info("add provider", "name", provider)
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
	slog.Info("deleteProvider", "name", provider)
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
	onStreamTextData func(textData string) error,
	onStreamThinkingData func(thinkingData string) error,
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
		onStreamTextData,
		onStreamThinkingData,
	)
	if err != nil {
		return nil, errors.Join(err, errors.New("error in fetch completion"))
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

func getProviderAPI(p spec.ProviderParam, debug bool) (spec.CompletionProvider, error) {
	switch p.SDKType {
	case spec.ProviderSDKTypeAnthropic:
		return anthropicsdk.NewAnthropicMessagesAPI(p, debug)

	case spec.ProviderSDKTypeOpenAIChatCompletions:
		return openaichatsdk.NewOpenAIChatCompletionsAPI(p, debug)

	case spec.ProviderSDKTypeOpenAIResponses:
		return openairesponsessdk.NewOpenAIResponsesAPI(p, debug)
	}

	return nil, errors.New("invalid provider api type")
}
