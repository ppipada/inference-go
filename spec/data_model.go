package spec

type (
	ModelName       string
	ReasoningLevel  string
	ReasoningType   string
	ProviderName    string
	ProviderSDKType string
)

const (
	ReasoningTypeHybridWithTokens ReasoningType = "hybridWithTokens"
	ReasoningTypeSingleWithLevels ReasoningType = "singleWithLevels"
)

const (
	ReasoningLevelNone    ReasoningLevel = "none"
	ReasoningLevelMinimal ReasoningLevel = "minimal"
	ReasoningLevelLow     ReasoningLevel = "low"
	ReasoningLevelMedium  ReasoningLevel = "medium"
	ReasoningLevelHigh    ReasoningLevel = "high"
	ReasoningLevelXHigh   ReasoningLevel = "xhigh"
)

type ReasoningParam struct {
	Type   ReasoningType  `json:"type"`
	Level  ReasoningLevel `json:"level"`
	Tokens int            `json:"tokens"`
}

type Usage struct {
	InputTokensTotal    int64 `json:"inputTokensTotal"`
	InputTokensCached   int64 `json:"inputTokensCached"`
	InputTokensUncached int64 `json:"inputTokensUncached"`
	OutputTokens        int64 `json:"outputTokens"`
	ReasoningTokens     int64 `json:"reasoningTokens"`
}

type ModelParam struct {
	Name                        ModelName       `json:"name"`
	Stream                      bool            `json:"stream"`
	MaxPromptLength             int             `json:"maxPromptLength"`
	MaxOutputLength             int             `json:"maxOutputLength"`
	Temperature                 *float64        `json:"temperature,omitempty"`
	Reasoning                   *ReasoningParam `json:"reasoning,omitempty"`
	SystemPrompt                string          `json:"systemPrompt"`
	Timeout                     int             `json:"timeout"`
	AdditionalParametersRawJSON *string         `json:"additionalParametersRawJSON"`
}
