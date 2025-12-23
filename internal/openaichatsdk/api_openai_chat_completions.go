package openaichatsdk

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
	openaiSharedConstant "github.com/openai/openai-go/v3/shared/constant"

	"github.com/ppipada/inference-go/internal/logutil"
	"github.com/ppipada/inference-go/internal/sdkutil"
	"github.com/ppipada/inference-go/spec"
)

// OpenAIChatCompletionsAPI struct that implements the CompletionProvider interface.
type OpenAIChatCompletionsAPI struct {
	ProviderParam *spec.ProviderParam

	debugger spec.CompletionDebugger
	client   *openai.Client
}

func NewOpenAIChatCompletionsAPI(
	pi spec.ProviderParam,
	debugger spec.CompletionDebugger,
) (*OpenAIChatCompletionsAPI, error) {
	if pi.Name == "" || pi.Origin == "" {
		return nil, errors.New("openai chat completions api LLM: invalid args")
	}
	return &OpenAIChatCompletionsAPI{
		ProviderParam: &pi,
		debugger:      debugger,
	}, nil
}

func (api *OpenAIChatCompletionsAPI) InitLLM(ctx context.Context) error {
	if !api.IsConfigured(ctx) {
		logutil.Debug(
			string(
				api.ProviderParam.Name,
			) + ": No API key given. Not initializing OpenAIChatCompletionsAPI LLM object",
		)
		return nil
	}

	opts := []option.RequestOption{
		option.WithAPIKey(api.ProviderParam.APIKey),
	}

	providerURL := spec.DefaultOpenAIOrigin
	if api.ProviderParam.Origin != "" {
		baseURL := strings.TrimSuffix(api.ProviderParam.Origin, "/")

		pathPrefix := api.ProviderParam.ChatCompletionPathPrefix
		// Remove "chat/completions" from pathPrefix if present; SDK adds it internally.
		pathPrefix = strings.TrimSuffix(
			pathPrefix,
			"chat/completions",
		)
		providerURL = baseURL + pathPrefix
		opts = append(opts, option.WithBaseURL(strings.TrimSuffix(providerURL, "/")))
	}

	for k, v := range api.ProviderParam.DefaultHeaders {
		opts = append(opts, option.WithHeader(strings.TrimSpace(k), strings.TrimSpace(v)))
	}

	if api.ProviderParam.APIKeyHeaderKey != "" &&
		!strings.EqualFold(
			api.ProviderParam.APIKeyHeaderKey,
			spec.DefaultAuthorizationHeaderKey,
		) {
		opts = append(
			opts,
			option.WithHeader(api.ProviderParam.APIKeyHeaderKey, api.ProviderParam.APIKey),
		)
	}

	if api.debugger != nil {
		if httpClient := api.debugger.HTTPClient(); httpClient != nil {
			opts = append(opts, option.WithHTTPClient(httpClient))
		}
	}

	c := openai.NewClient(opts...)
	api.client = &c
	logutil.Info(
		"openai chat completions api LLM provider initialized",
		"name",
		string(api.ProviderParam.Name),
		"URL",
		providerURL,
	)
	return nil
}

func (api *OpenAIChatCompletionsAPI) DeInitLLM(ctx context.Context) error {
	api.client = nil
	logutil.Info(
		"openai chat completions api LLM: provider de initialized",
		"name",
		string(api.ProviderParam.Name),
	)
	return nil
}

func (api *OpenAIChatCompletionsAPI) GetProviderInfo(ctx context.Context) *spec.ProviderParam {
	return api.ProviderParam
}

func (api *OpenAIChatCompletionsAPI) IsConfigured(ctx context.Context) bool {
	return api.ProviderParam != nil && api.ProviderParam.APIKey != ""
}

// SetProviderAPIKey sets the key for a provider.
func (api *OpenAIChatCompletionsAPI) SetProviderAPIKey(
	ctx context.Context,
	apiKey string,
) error {
	if apiKey == "" {
		return errors.New("openai chat completions api LLM: invalid apikey provided")
	}
	if api.ProviderParam == nil {
		return errors.New("openai chat completions api LLM: no ProviderParam found")
	}

	api.ProviderParam.APIKey = apiKey

	return nil
}

func (api *OpenAIChatCompletionsAPI) FetchCompletion(
	ctx context.Context,
	req *spec.FetchCompletionRequest,
	opts *spec.FetchCompletionOptions,
) (*spec.FetchCompletionResponse, error) {
	if api.client == nil {
		return nil, errors.New("openai chat completions api LLM: client not initialized")
	}
	if req == nil || len(req.Inputs) == 0 || req.ModelParam.Name == "" {
		return nil, errors.New("openai chat completions api LLM: empty completion data")
	}

	// Build OpenAI chat messages.
	msgs, err := toOpenAIChatMessages(
		ctx,
		req.ModelParam.SystemPrompt,
		req.Inputs,
		req.ModelParam.Name,
		api.ProviderParam.Name,
	)
	if err != nil {
		return nil, err
	}

	params := openai.ChatCompletionNewParams{
		Model:               shared.ChatModel(req.ModelParam.Name),
		MaxCompletionTokens: openai.Int(int64(req.ModelParam.MaxOutputLength)),
		Messages:            msgs,
	}
	if t := req.ModelParam.Temperature; t != nil {
		params.Temperature = openai.Float(*t)
	}

	if rp := req.ModelParam.Reasoning; rp != nil &&
		rp.Type == spec.ReasoningTypeSingleWithLevels {
		switch rp.Level {
		case
			spec.ReasoningLevelNone,
			spec.ReasoningLevelMinimal,
			spec.ReasoningLevelLow,
			spec.ReasoningLevelMedium,
			spec.ReasoningLevelHigh,
			spec.ReasoningLevelXHigh:
			params.ReasoningEffort = shared.ReasoningEffort(string(rp.Level))
		default:
			return nil, fmt.Errorf("invalid level %q for singleWithLevels", rp.Level)

		}
	}
	var toolChoiceNameMap map[string]spec.ToolChoice
	if len(req.ToolChoices) > 0 {
		toolDefs, nameMap, err := toolChoicesToOpenAIChatTools(req.ToolChoices)
		if err != nil {
			return nil, err
		}
		if len(toolDefs) > 0 {
			params.Tools = toolDefs
			toolChoiceNameMap = nameMap
		}
		// Map a single webSearch ToolChoice (if any) to top-level web_search_options.
		if ws := firstWebSearchToolChoice(req.ToolChoices); ws != nil && ws.WebSearchArguments != nil {
			applyOpenAIChatWebSearchOptions(&params, ws.WebSearchArguments)
		}
	}

	timeout := spec.DefaultAPITimeout
	if req.ModelParam.Timeout > 0 {
		timeout = time.Duration(req.ModelParam.Timeout) * time.Second
	}
	if api.debugger != nil {
		ctx = api.debugger.WrapContext(ctx)
	}

	useStream := req.ModelParam.Stream && opts != nil && opts.StreamHandler != nil
	if useStream {
		return api.doStreaming(ctx, req.ModelParam.Name, params, opts, timeout, toolChoiceNameMap)
	}
	return api.doNonStreaming(ctx, params, timeout, toolChoiceNameMap)
}

func (api *OpenAIChatCompletionsAPI) doNonStreaming(
	ctx context.Context,
	params openai.ChatCompletionNewParams,
	timeout time.Duration,
	toolChoiceNameMap map[string]spec.ToolChoice,
) (*spec.FetchCompletionResponse, error) {
	resp := &spec.FetchCompletionResponse{}

	oaiResp, err := api.client.Chat.Completions.New(ctx, params, option.WithRequestTimeout(timeout))

	isNilResp := oaiResp == nil || len(oaiResp.Choices) == 0
	if api.debugger != nil {
		resp.DebugDetails = api.debugger.BuildDebugDetails(ctx, oaiResp, err, isNilResp)
	}
	resp.Usage = usageFromOpenAIChatCompletion(oaiResp)
	if err != nil {
		resp.Error = &spec.Error{Message: err.Error()}
		return resp, err
	}

	if !isNilResp {
		resp.Outputs = outputsFromOpenAIChatCompletion(oaiResp, toolChoiceNameMap)
	}
	return resp, nil
}

func (api *OpenAIChatCompletionsAPI) doStreaming(
	ctx context.Context,
	modelName spec.ModelName,
	params openai.ChatCompletionNewParams,
	opts *spec.FetchCompletionOptions,
	timeout time.Duration,
	toolChoiceNameMap map[string]spec.ToolChoice,
) (*spec.FetchCompletionResponse, error) {
	resp := &spec.FetchCompletionResponse{}
	streamCfg := sdkutil.ResolveStreamConfig(opts)
	// No thinking data available in openai chat completions API, hence no thinking writer.
	emitText := func(chunk string) error {
		if strings.TrimSpace(chunk) == "" {
			return nil
		}
		event := spec.StreamEvent{
			Kind:     spec.StreamContentKindText,
			Provider: api.ProviderParam.Name,
			Model:    modelName,
			Text: &spec.StreamTextChunk{
				Text: chunk,
			},
		}
		return sdkutil.SafeCallStreamHandler(opts.StreamHandler, event)
	}

	// No thinking data available in openai chat completions API, hence no thinking writer.
	writeText, flushText := sdkutil.NewBufferedStreamer(
		emitText,
		streamCfg.FlushInterval,
		streamCfg.FlushChunkSize,
	)

	stream := api.client.Chat.Completions.NewStreaming(
		ctx,
		params,
		option.WithRequestTimeout(timeout),
	)
	defer func() { _ = stream.Close() }()

	acc := openai.ChatCompletionAccumulator{}
	var streamWriteErr error
	for stream.Next() {
		chunk := stream.Current()
		acc.AddChunk(chunk)

		// When JustFinished* triggers, the current chunk isn't textual content.
		if _, ok := acc.JustFinishedContent(); ok {
			continue
		}

		if _, ok := acc.JustFinishedRefusal(); ok {
			continue
		}

		if _, ok := acc.JustFinishedToolCall(); ok {
			continue
		}

		// Best to use chunks after handling JustFinished events.
		if len(chunk.Choices) > 0 && strings.TrimSpace(chunk.Choices[0].Delta.Content) != "" {
			streamWriteErr = writeText(chunk.Choices[0].Delta.Content)
			if streamWriteErr != nil {
				break
			}
		}
	}
	if flushText != nil {
		flushText()
	}

	streamErr := errors.Join(stream.Err(), streamWriteErr)
	isNilResp := len(acc.Choices) == 0

	if api.debugger != nil {
		resp.DebugDetails = api.debugger.BuildDebugDetails(ctx, &acc.ChatCompletion, streamErr, isNilResp)
	}

	resp.Usage = usageFromOpenAIChatCompletion(&acc.ChatCompletion)
	if streamErr != nil {
		resp.Error = &spec.Error{Message: streamErr.Error()}
	}

	if !isNilResp {
		resp.Outputs = outputsFromOpenAIChatCompletion(&acc.ChatCompletion, toolChoiceNameMap)
	}

	return resp, streamErr
}

func toOpenAIChatMessages(
	_ context.Context,
	systemPrompt string,
	inputs []spec.InputUnion,
	modelName spec.ModelName,
	providerName spec.ProviderName,
) ([]openai.ChatCompletionMessageParamUnion, error) {
	var out []openai.ChatCompletionMessageParamUnion

	// Top-level system/developer instructions.
	if msg := getOpenAIMessageFromSystemPrompt(providerName, modelName, systemPrompt); msg != nil {
		out = append(out, *msg)
	}

	for _, in := range inputs {
		if sdkutil.IsInputUnionEmpty(in) {
			continue
		}

		switch in.Kind {

		case spec.InputKindInputMessage:
			// Only user role is valid for InputMessage here; dev/system handled via systemPrompt.
			if in.InputMessage == nil || in.InputMessage.Role != spec.RoleUser {
				continue
			}
			parts, err := contentItemsToOpenAIUserMessageParts(in.InputMessage.Contents)
			if err != nil {
				return nil, err
			}
			if len(parts) > 0 {
				out = append(out, openai.UserMessage(parts))
			}

		case spec.InputKindOutputMessage:
			// Assistant prior text outputs become assistant messages.
			if in.OutputMessage == nil || in.OutputMessage.Role != spec.RoleAssistant {
				continue
			}
			parts := contentItemsToAssistantMessageParts(in.OutputMessage.Contents)
			if len(parts) > 0 {
				out = append(out, openai.AssistantMessage(parts))
			}

		case spec.InputKindFunctionToolCall, spec.InputKindCustomToolCall:
			var call *spec.ToolCall
			if in.FunctionToolCall != nil {
				call = in.FunctionToolCall
			} else if in.CustomToolCall != nil {
				call = in.CustomToolCall
			}
			if m := toolCallToOpenAIChatAssistantMessage(call); m != nil {
				out = append(out, *m)
			}

		case spec.InputKindFunctionToolOutput, spec.InputKindCustomToolOutput:
			var output *spec.ToolOutput
			if in.FunctionToolOutput != nil {
				output = in.FunctionToolOutput
			} else if in.CustomToolOutput != nil {
				output = in.CustomToolOutput
			}
			if m := toolOutputToOpenAIChatMessages(output); m != nil {
				out = append(out, *m)
			}

		case spec.InputKindReasoningMessage:
			// Chat Completions has no structured reasoning messages.
			continue

		case spec.InputKindWebSearchToolCall, spec.InputKindWebSearchToolOutput:
			// Chat Completions doesn't expose web search as a tool;
			// it is configured via top-level web_search_options instead.
			continue
		}
	}

	return out, nil
}

func contentItemsToOpenAIUserMessageParts(
	items []spec.InputOutputContentItemUnion,
) ([]openai.ChatCompletionContentPartUnionParam, error) {
	out := make([]openai.ChatCompletionContentPartUnionParam, 0, len(items))

	for _, it := range items {
		switch it.Kind {
		case spec.ContentItemKindText:
			if it.TextItem == nil {
				continue
			}
			txt := strings.TrimSpace(it.TextItem.Text)
			if txt == "" {
				continue
			}
			out = append(out, openai.TextContentPart(txt))

		case spec.ContentItemKindImage:
			if it.ImageItem == nil {
				continue
			}
			img := it.ImageItem

			// Prefer embedded base64 data if present.
			if data := strings.TrimSpace(img.ImageData); data != "" {
				mime := strings.TrimSpace(img.ImageMIME)
				if mime == "" {
					mime = spec.DefaultImageDataMIME
				}
				dataURL := fmt.Sprintf("data:%s;base64,%s", mime, data)
				part := openai.ChatCompletionContentPartImageImageURLParam{
					URL:    dataURL,
					Detail: string(img.Detail),
				}
				out = append(out, openai.ImageContentPart(part))

			} else if u := strings.TrimSpace(img.ImageURL); u != "" {
				part := openai.ChatCompletionContentPartImageImageURLParam{
					URL:    u,
					Detail: string(img.Detail),
				}
				out = append(out, openai.ImageContentPart(part))
			}

		case spec.ContentItemKindFile:
			if it.FileItem == nil {
				continue
			}
			f := it.FileItem

			// Embedded data as data URL.
			// Chat completions doesn't support sending file URL only.
			if data := strings.TrimSpace(f.FileData); data != "" {
				mime := strings.TrimSpace(f.FileMIME)
				if mime == "" {
					mime = spec.DefaultFileDataMIME
				}
				dataURL := fmt.Sprintf("data:%s;base64,%s", mime, data)
				var fileParam openai.ChatCompletionContentPartFileFileParam
				fileParam.FileData = param.NewOpt(dataURL)
				if name := strings.TrimSpace(f.FileName); name != "" {
					fileParam.Filename = param.NewOpt(name)
				}
				out = append(out, openai.FileContentPart(fileParam))

			}

		case spec.ContentItemKindRefusal:
			// Refusals are assistant outputs, not user inputs.
			continue

		default:
			logutil.Debug("chat completions: unknown content item kind for input message", "kind", it.Kind)
		}
	}

	return out, nil
}

func contentItemsToAssistantMessageParts(
	items []spec.InputOutputContentItemUnion,
) []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion {
	if len(items) == 0 {
		return nil
	}
	parts := make([]openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion, 0)
	addedRefusal := false

	for _, it := range items {
		switch it.Kind {
		case spec.ContentItemKindText:
			if it.TextItem != nil {
				if s := strings.TrimSpace(it.TextItem.Text); s != "" {
					parts = append(parts, openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
						OfText: &openai.ChatCompletionContentPartTextParam{
							Text: s,
						},
					})
				}
			}
		case spec.ContentItemKindRefusal:
			if addedRefusal {
				// Chat completions needs only one refusal objet, if at all.
				continue
			}
			if it.RefusalItem != nil {
				if s := strings.TrimSpace(it.RefusalItem.Refusal); s != "" {
					parts = append(parts, openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
						OfRefusal: &openai.ChatCompletionContentPartRefusalParam{
							Refusal: s,
						},
					})
					addedRefusal = true
				}
			}
		default:
			// No image or file support in chat completions.
		}
	}
	return parts
}

func toolCallToOpenAIChatAssistantMessage(
	call *spec.ToolCall,
) *openai.ChatCompletionMessageParamUnion {
	if call == nil || strings.TrimSpace(call.ID) == "" {
		return nil
	}

	switch call.Type {
	case spec.ToolTypeFunction:
		f := openai.ChatCompletionMessageFunctionToolCallParam{
			ID: call.ID,
			Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
				Name:      call.Name,
				Arguments: call.Arguments,
			},
			Type: openaiSharedConstant.Function("").Default(),
		}
		msg := openai.ChatCompletionAssistantMessageParam{
			Role: openaiSharedConstant.Assistant("").Default(),
			ToolCalls: []openai.ChatCompletionMessageToolCallUnionParam{
				{OfFunction: &f},
			},
		}
		return &openai.ChatCompletionMessageParamUnion{OfAssistant: &msg}

	case spec.ToolTypeCustom:
		c := openai.ChatCompletionMessageCustomToolCallParam{
			ID: call.ID,
			Custom: openai.ChatCompletionMessageCustomToolCallCustomParam{
				Name:  call.Name,
				Input: call.Arguments,
			},
			Type: openaiSharedConstant.Custom("").Default(),
		}
		msg := openai.ChatCompletionAssistantMessageParam{
			Role: openaiSharedConstant.Assistant("").Default(),
			ToolCalls: []openai.ChatCompletionMessageToolCallUnionParam{
				{OfCustom: &c},
			},
		}
		return &openai.ChatCompletionMessageParamUnion{OfAssistant: &msg}
	case spec.ToolTypeWebSearch:
		// WebSearch is not represented as tool call in Chat Completions.
	}
	return nil
}

func toolOutputToOpenAIChatMessages(
	output *spec.ToolOutput,
) *openai.ChatCompletionMessageParamUnion {
	if output == nil || strings.TrimSpace(output.CallID) == "" || len(output.Contents) == 0 {
		return nil
	}

	parts := make([]openai.ChatCompletionContentPartTextParam, 0)
	for _, it := range output.Contents {
		if it.Kind == spec.ContentItemKindText && it.TextItem != nil {
			if s := strings.TrimSpace(it.TextItem.Text); s != "" {
				parts = append(parts, openai.ChatCompletionContentPartTextParam{
					Text: s,
				})
			}
		}
	}

	msg := openai.ToolMessage(parts, output.CallID)
	return &msg
}

// getOpenAIMessageFromSystemPrompt returns a single system/developer message
// based on the model and provider.
func getOpenAIMessageFromSystemPrompt(
	providerName spec.ProviderName,
	modelName spec.ModelName,
	systemPrompt string,
) *openai.ChatCompletionMessageParamUnion {
	sp := strings.TrimSpace(systemPrompt)
	if sp == "" {
		return nil
	}
	msg := openai.SystemMessage(sp)
	// Convert a system message to a developer message for o* / gpt-5* models.
	if providerName == "openai" &&
		(strings.HasPrefix(string(modelName), "o") ||
			strings.HasPrefix(string(modelName), "gpt-5")) {
		msg = openai.DeveloperMessage(sp)
	}
	return &msg
}

func toolChoicesToOpenAIChatTools(
	toolChoices []spec.ToolChoice,
) ([]openai.ChatCompletionToolUnionParam, map[string]spec.ToolChoice, error) {
	if len(toolChoices) == 0 {
		return []openai.ChatCompletionToolUnionParam{}, nil, nil
	}

	ordered, nameMap := sdkutil.BuildToolChoiceNameMapping(toolChoices)
	out := make([]openai.ChatCompletionToolUnionParam, 0, len(ordered))

	for _, tw := range ordered {
		tc := tw.Choice
		name := tw.Name

		switch tc.Type {
		case spec.ToolTypeFunction, spec.ToolTypeCustom:
			if tc.Arguments == nil || name == "" {
				continue
			}
			// For now, both function and custom tools are expressed as function tools,
			// mirroring the Responses adapter behavior.
			fn := shared.FunctionDefinitionParam{
				Name:       name,
				Parameters: tc.Arguments,
			}
			if desc := sdkutil.ToolDescription(tc); desc != "" {
				fn.Description = openai.String(desc)
			}
			out = append(out, openai.ChatCompletionFunctionTool(fn))

		case spec.ToolTypeWebSearch:
			// Web search is not exposed as a Chat Completions tool; handled via top-level web_search_options instead.
			continue
		}
	}

	if len(out) == 0 {
		return []openai.ChatCompletionToolUnionParam{}, nil, nil
	}
	return out, nameMap, nil
}

// firstWebSearchToolChoice returns the first ToolChoice of type webSearch, if any.
func firstWebSearchToolChoice(tools []spec.ToolChoice) *spec.ToolChoice {
	for i := range tools {
		if tools[i].Type == spec.ToolTypeWebSearch && tools[i].WebSearchArguments != nil {
			return &tools[i]
		}
	}
	return nil
}

func applyOpenAIChatWebSearchOptions(
	params *openai.ChatCompletionNewParams,
	ws *spec.WebSearchToolChoiceItem,
) {
	if params == nil || ws == nil {
		return
	}

	var opt openai.ChatCompletionNewParamsWebSearchOptions

	searchContextSize := strings.ToLower(strings.TrimSpace(ws.SearchContextSize))
	switch searchContextSize {
	case "low", "medium", "high":
		opt.SearchContextSize = searchContextSize
	default:
		// Default to "medium" if unset/invalid.
		opt.SearchContextSize = "medium"
	}

	if ws.UserLocation != nil {
		approx := openai.ChatCompletionNewParamsWebSearchOptionsUserLocationApproximate{}
		if s := strings.TrimSpace(ws.UserLocation.City); s != "" {
			approx.City = openai.String(s)
		}
		if s := strings.TrimSpace(ws.UserLocation.Country); s != "" {
			approx.Country = openai.String(s)
		}
		if s := strings.TrimSpace(ws.UserLocation.Region); s != "" {
			approx.Region = openai.String(s)
		}
		if s := strings.TrimSpace(ws.UserLocation.Timezone); s != "" {
			approx.Timezone = openai.String(s)
		}
		opt.UserLocation = openai.ChatCompletionNewParamsWebSearchOptionsUserLocation{
			Approximate: approx,
		}
	}

	params.WebSearchOptions = opt
}

func outputsFromOpenAIChatCompletion(
	resp *openai.ChatCompletion,
	toolChoiceNameMap map[string]spec.ToolChoice,
) []spec.OutputUnion {
	if resp == nil || len(resp.Choices) == 0 {
		return nil
	}

	choice := resp.Choices[0]
	msg := choice.Message
	status := mapOpenAIChatFinishReasonToStatus(choice.FinishReason)

	var outs []spec.OutputUnion

	// Assistant text output.
	if refusal := strings.TrimSpace(msg.Refusal); refusal != "" {
		refusalItem := spec.ContentItemRefusal{
			Refusal: refusal,
		}

		outMsg := spec.InputOutputContent{
			ID:   resp.ID,
			Role: spec.RoleAssistant,
			// Chat Completions does not expose per-block status; use finish_reason.
			Status: status,
			Contents: []spec.InputOutputContentItemUnion{{
				Kind:        spec.ContentItemKindRefusal,
				RefusalItem: &refusalItem,
			}},
		}
		outs = append(
			outs,
			spec.OutputUnion{
				Kind:          spec.OutputKindOutputMessage,
				OutputMessage: &outMsg,
			},
		)
	} else if txt := strings.TrimSpace(msg.Content); txt != "" {
		textItem := spec.ContentItemText{
			Text: txt,
		}
		if len(msg.Annotations) > 0 {
			textItem.Citations = chatAnnotationsToCitations(msg.Annotations)
		}

		outMsg := spec.InputOutputContent{
			ID:   resp.ID,
			Role: spec.RoleAssistant,
			// Chat Completions does not expose per-block status; use finish_reason.
			Status: status,
			Contents: []spec.InputOutputContentItemUnion{{
				Kind:     spec.ContentItemKindText,
				TextItem: &textItem,
			}},
		}
		outs = append(
			outs,
			spec.OutputUnion{
				Kind:          spec.OutputKindOutputMessage,
				OutputMessage: &outMsg,
			},
		)
	}

	// Tool calls (function/custom).
	if len(msg.ToolCalls) > 0 {
		for _, tc := range msg.ToolCalls {
			switch tc.Type {
			case string(openaiSharedConstant.Function("").Default()):
				if tc.ID == "" || strings.TrimSpace(tc.Function.Name) == "" {
					continue
				}
				name := tc.Function.Name
				var choiceID string
				if toolChoiceNameMap != nil {
					if tcDef, ok := toolChoiceNameMap[name]; ok {
						choiceID = tcDef.ID
					}
				}
				if choiceID == "" {
					continue
				}
				call := spec.ToolCall{
					ChoiceID:  choiceID,
					Type:      spec.ToolTypeFunction,
					Role:      spec.RoleAssistant,
					ID:        tc.ID,
					CallID:    tc.ID,
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
					Status:    status,
				}
				outs = append(
					outs,
					spec.OutputUnion{
						Kind:             spec.OutputKindFunctionToolCall,
						FunctionToolCall: &call,
					},
				)

			case string(openaiSharedConstant.Custom("").Default()):
				if tc.ID == "" || strings.TrimSpace(tc.Custom.Name) == "" {
					continue
				}
				name := tc.Custom.Name
				var choiceID string
				if toolChoiceNameMap != nil {
					if tcDef, ok := toolChoiceNameMap[name]; ok {
						choiceID = tcDef.ID
					}
				}
				if choiceID == "" {
					continue
				}
				call := spec.ToolCall{
					ChoiceID:  choiceID,
					Type:      spec.ToolTypeCustom,
					Role:      spec.RoleAssistant,
					ID:        tc.ID,
					CallID:    tc.ID,
					Name:      tc.Custom.Name,
					Arguments: tc.Custom.Input,
					// No explicit status for custom tool calls; treat as completed.
					Status: spec.StatusCompleted,
				}
				outs = append(
					outs,
					spec.OutputUnion{
						Kind:           spec.OutputKindCustomToolCall,
						CustomToolCall: &call,
					},
				)
			}
		}
	}

	if len(outs) == 0 {
		return nil
	}
	return outs
}

func chatAnnotationsToCitations(
	anns []openai.ChatCompletionMessageAnnotation,
) []spec.Citation {
	if len(anns) == 0 {
		return nil
	}
	out := make([]spec.Citation, 0)
	for _, a := range anns {
		if string(a.Type) != string(openaiSharedConstant.URLCitation("").Default()) {
			// Only URL citations are currently supported.
			continue
		}
		out = append(out, spec.Citation{
			Kind: spec.CitationKindURL,
			URLCitation: &spec.URLCitation{
				URL:        a.URLCitation.URL,
				Title:      a.URLCitation.Title,
				StartIndex: a.URLCitation.StartIndex,
				EndIndex:   a.URLCitation.EndIndex,
			},
		})
	}
	return out
}

func mapOpenAIChatFinishReasonToStatus(reason string) spec.Status {
	switch reason {
	case "length":
		return spec.StatusIncomplete
	case "content_filter":
		return spec.StatusFailed
	case "stop", "tool_calls":
		return spec.StatusCompleted
	default:
		// Treat unknown/empty as completed; HTTP error will be surfaced separately.
		return spec.StatusCompleted
	}
}

func usageFromOpenAIChatCompletion(resp *openai.ChatCompletion) *spec.Usage {
	uOut := &spec.Usage{}
	if resp == nil {
		return uOut
	}

	u := resp.Usage

	uOut.InputTokensTotal = u.PromptTokens
	uOut.InputTokensCached = u.PromptTokensDetails.CachedTokens
	uOut.InputTokensUncached = max(u.PromptTokens-u.PromptTokensDetails.CachedTokens, 0)
	uOut.OutputTokens = u.CompletionTokens
	uOut.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens

	return uOut
}
