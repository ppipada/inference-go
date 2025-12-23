package openairesponsessdk

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	openaiSharedConstant "github.com/openai/openai-go/v3/shared/constant"

	"github.com/ppipada/inference-go/internal/logutil"
	"github.com/ppipada/inference-go/internal/sdkutil"
	"github.com/ppipada/inference-go/spec"
)

// OpenAIResponsesAPI struct that implements the CompletionProvider interface.
type OpenAIResponsesAPI struct {
	ProviderParam *spec.ProviderParam

	debugger spec.CompletionDebugger

	client *openai.Client
}

func NewOpenAIResponsesAPI(
	pi spec.ProviderParam,
	debugger spec.CompletionDebugger,
) (*OpenAIResponsesAPI, error) {
	if pi.Name == "" || pi.Origin == "" {
		return nil, errors.New("openai responses api LLM: invalid args")
	}
	return &OpenAIResponsesAPI{
		ProviderParam: &pi,
		debugger:      debugger,
	}, nil
}

func (api *OpenAIResponsesAPI) InitLLM(ctx context.Context) error {
	if !api.IsConfigured(ctx) {
		logutil.Debug(
			string(
				api.ProviderParam.Name,
			) + ": No API key given. Not initializing OpenAIResponsesAPI LLM object",
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
		// Remove "responses" from pathPrefix if present; SDK adds it internally.
		pathPrefix = strings.TrimSuffix(pathPrefix, "responses")

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
		"openai responses api LLM provider initialized",
		"name",
		string(api.ProviderParam.Name),
		"URL",
		providerURL,
	)
	return nil
}

func (api *OpenAIResponsesAPI) DeInitLLM(ctx context.Context) error {
	api.client = nil
	logutil.Info(
		"openai responses api LLM: provider de initialized",
		"name",
		string(api.ProviderParam.Name),
	)
	return nil
}

func (api *OpenAIResponsesAPI) GetProviderInfo(ctx context.Context) *spec.ProviderParam {
	return api.ProviderParam
}

func (api *OpenAIResponsesAPI) IsConfigured(ctx context.Context) bool {
	return api.ProviderParam != nil && api.ProviderParam.APIKey != ""
}

// SetProviderAPIKey sets the key for a provider.
func (api *OpenAIResponsesAPI) SetProviderAPIKey(
	ctx context.Context,
	apiKey string,
) error {
	if apiKey == "" {
		return errors.New("openai responses api LLM: invalid apikey provided")
	}
	if api.ProviderParam == nil {
		return errors.New("openai responses api LLM: no ProviderParam found")
	}

	api.ProviderParam.APIKey = apiKey

	return nil
}

func (api *OpenAIResponsesAPI) FetchCompletion(
	ctx context.Context,
	req *spec.FetchCompletionRequest,
	opts *spec.FetchCompletionOptions,
) (*spec.FetchCompletionResponse, error) {
	if api.client == nil {
		return nil, errors.New("openai responses api LLM: client not initialized")
	}
	if req == nil || len(req.Inputs) == 0 || req.ModelParam.Name == "" {
		return nil, errors.New("openai responses api LLM: invalid data")
	}

	// Build OpenAI Responses input messages.
	inputItems, err := toOpenAIResponsesInput(
		ctx,
		req.Inputs,
	)
	if err != nil {
		return nil, err
	}

	params := responses.ResponseNewParams{
		Model:           shared.ChatModel(req.ModelParam.Name),
		MaxOutputTokens: openai.Int(int64(req.ModelParam.MaxOutputLength)),
		Input:           responses.ResponseNewParamsInputUnion{OfInputItemList: inputItems},
		Store:           openai.Bool(false),
		Include:         []responses.ResponseIncludable{"reasoning.encrypted_content"},
	}

	// Top‑level instructions.
	if sys := strings.TrimSpace(req.ModelParam.SystemPrompt); sys != "" {
		params.Instructions = openai.String(sys)
	}
	if req.ModelParam.Temperature != nil {
		params.Temperature = openai.Float(*req.ModelParam.Temperature)
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
			params.Reasoning = shared.ReasoningParam{
				Effort:  shared.ReasoningEffort(string(rp.Level)),
				Summary: shared.ReasoningSummaryAuto,
			}
		default:
			return nil, fmt.Errorf("invalid reasoning level %q for singleWithLevels", rp.Level)
		}
	}

	var toolChoiceNameMap map[string]spec.ToolChoice
	if len(req.ToolChoices) > 0 {
		toolDefs, nameMap, err := toolChoicesToOpenAIResponseTools(req.ToolChoices)
		if err != nil {
			return nil, err
		}
		if len(toolDefs) > 0 {
			params.Tools = toolDefs
			toolChoiceNameMap = nameMap
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

func (api *OpenAIResponsesAPI) doNonStreaming(
	ctx context.Context,
	params responses.ResponseNewParams,
	timeout time.Duration,
	toolChoiceNameMap map[string]spec.ToolChoice,
) (*spec.FetchCompletionResponse, error) {
	resp := &spec.FetchCompletionResponse{}

	oaiResp, err := api.client.Responses.New(ctx, params, option.WithRequestTimeout(timeout))
	isNilResp := oaiResp == nil || len(oaiResp.Output) == 0
	if api.debugger != nil {
		resp.DebugDetails = api.debugger.BuildDebugDetails(ctx, oaiResp, err, isNilResp)
	}
	resp.Usage = usageFromOpenAIResponse(oaiResp)

	if err != nil {
		resp.Error = &spec.Error{Message: err.Error()}
		// Even on error, return any partial usage/debug we have.
		return resp, err
	}

	resp.Outputs = outputsFromOpenAIResponse(oaiResp, toolChoiceNameMap)
	return resp, nil
}

func (api *OpenAIResponsesAPI) doStreaming(
	ctx context.Context,
	modelName spec.ModelName,
	params responses.ResponseNewParams,
	opts *spec.FetchCompletionOptions,
	timeout time.Duration,
	toolChoiceNameMap map[string]spec.ToolChoice,
) (*spec.FetchCompletionResponse, error) {
	resp := &spec.FetchCompletionResponse{}
	streamCfg := sdkutil.ResolveStreamConfig(opts)

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

	emitThinking := func(chunk string) error {
		if strings.TrimSpace(chunk) == "" {
			return nil
		}
		event := spec.StreamEvent{
			Kind:     spec.StreamContentKindThinking,
			Provider: api.ProviderParam.Name,
			Model:    modelName,
			Thinking: &spec.StreamThinkingChunk{
				Text: chunk,
			},
		}
		return sdkutil.SafeCallStreamHandler(opts.StreamHandler, event)
	}

	writeTextData, flushTextData := sdkutil.NewBufferedStreamer(
		emitText,
		streamCfg.FlushInterval,
		streamCfg.FlushChunkSize,
	)
	writeThinkingData, flushThinkingData := sdkutil.NewBufferedStreamer(
		emitThinking,
		streamCfg.FlushInterval,
		streamCfg.FlushChunkSize,
	)

	var respFull responses.Response

	stream := api.client.Responses.NewStreaming(
		ctx,
		params,
		option.WithRequestTimeout(timeout),
	)
	defer func() { _ = stream.Close() }()

	var streamWriteErr error
	for stream.Next() {
		chunk := stream.Current()

		// Incremental assistant text.
		if chunk.Type == "response.output_text.delta" {
			streamWriteErr = writeTextData(chunk.Delta)
			if streamWriteErr != nil {
				break
			}
		}

		// Incremental reasoning text.
		if chunk.Type == "response.reasoning_summary_text.delta" {
			streamWriteErr = writeThinkingData(chunk.Delta)
			if streamWriteErr != nil {
				break
			}
		}

		// Incremental reasoning text.
		if chunk.Type == "response.reasoning_text.delta" {
			streamWriteErr = writeThinkingData(chunk.Delta)
			if streamWriteErr != nil {
				break
			}
		}

		if chunk.Type == "response.completed" {
			respFull = chunk.Response
			// Normal completion.
			break
		}

		if chunk.Type == "response.failed" {
			respFull = chunk.Response
			streamWriteErr = fmt.Errorf("API failed, %s", respFull.Error.RawJSON())
			break
		}

		if chunk.Type == "response.incomplete" {
			respFull = chunk.Response
			streamWriteErr = fmt.Errorf("API finished as incomplete, %s", respFull.IncompleteDetails.Reason)
			break
		}

	}
	if flushTextData != nil {
		flushTextData()
	}
	if flushThinkingData != nil {
		flushThinkingData()
	}

	streamErr := errors.Join(stream.Err(), streamWriteErr)
	isNilResp := len(respFull.Output) == 0

	if api.debugger != nil {
		resp.DebugDetails = api.debugger.BuildDebugDetails(ctx, &respFull, streamErr, isNilResp)
	}

	resp.Usage = usageFromOpenAIResponse(&respFull)
	if streamErr != nil {
		resp.Error = &spec.Error{Message: streamErr.Error()}
	}

	if len(respFull.Output) > 0 {
		resp.Outputs = outputsFromOpenAIResponse(&respFull, toolChoiceNameMap)
	}

	return resp, streamErr
}

func toOpenAIResponsesInput(
	_ context.Context,
	inputs []spec.InputUnion,
) (responses.ResponseInputParam, error) {
	var out responses.ResponseInputParam

	for _, in := range inputs {
		if sdkutil.IsInputUnionEmpty(in) {
			continue
		}

		switch in.Kind {
		case spec.InputKindInputMessage:
			if in.InputMessage == nil || in.InputMessage.Role != spec.RoleUser {
				// We do not send dev or system message internally.
				// That is via top level instructions field.
				// Other roles are not valid for input message type.
				continue
			}
			items, err := contentItemsToOpenAIInputContent(in.InputMessage.Contents)
			if err != nil {
				return nil, err
			}
			if len(items) == 0 {
				continue
			}

			out = append(out, responses.ResponseInputItemUnionParam{
				OfInputMessage: &responses.ResponseInputItemMessageParam{
					Content: items,
					Role:    string(responses.EasyInputMessageRoleUser),
					Status:  toOpenAIStatus(in.InputMessage.Status),
				},
			})

		case spec.InputKindOutputMessage:
			if in.OutputMessage == nil || in.OutputMessage.Role != spec.RoleAssistant {
				// We do not send any other output message other than output text and refusal.
				// Both are assistant generated.
				continue
			}
			items, err := contentItemsToOpenAIOutputContent(in.OutputMessage.Contents)
			if err != nil {
				return nil, err
			}
			if len(items) == 0 {
				continue
			}
			status := responses.ResponseOutputMessageStatusCompleted
			if in.OutputMessage.Status == fromOpenAIStatus(string(responses.ResponseOutputMessageStatusIncomplete)) {
				status = responses.ResponseOutputMessageStatusIncomplete
			} else if in.OutputMessage.Status == fromOpenAIStatus(string(responses.ResponseOutputMessageStatusInProgress)) {
				status = responses.ResponseOutputMessageStatusInProgress
			}

			out = append(out, responses.ResponseInputItemUnionParam{
				OfOutputMessage: &responses.ResponseOutputMessageParam{
					ID:      in.OutputMessage.ID,
					Content: items,
					Status:  status,
				},
			})

		case spec.InputKindReasoningMessage:
			if in.ReasoningMessage != nil {
				if item := reasoningContentToOpenAIItem(in.ReasoningMessage); item != nil {
					out = append(out, *item)
				}
			}

		case spec.InputKindFunctionToolCall, spec.InputKindCustomToolCall, spec.InputKindWebSearchToolCall:
			var call *spec.ToolCall
			switch {
			case in.FunctionToolCall != nil:
				call = in.FunctionToolCall
			case in.CustomToolCall != nil:
				call = in.CustomToolCall
			case in.WebSearchToolCall != nil:
				call = in.WebSearchToolCall
			}

			if tc := toolCallToOpenAIItem(call); tc != nil {
				out = append(out, *tc)
			}

		case spec.InputKindFunctionToolOutput, spec.InputKindCustomToolOutput:
			var output *spec.ToolOutput
			if in.FunctionToolOutput != nil {
				output = in.FunctionToolOutput
			} else if in.CustomToolOutput != nil {
				output = in.CustomToolOutput
			}

			if tc := toolOutputToOpenAIResponses(output); tc != nil {
				out = append(out, *tc)
			}

		case spec.InputKindWebSearchToolOutput:
			// Ok. Responses doesn't have a web search output.
		}
	}

	return out, nil
}

// contentItemsToOpenAI converts spec.Content items to OpenAI input message parts.
func contentItemsToOpenAIInputContent(
	items []spec.InputOutputContentItemUnion,
) ([]responses.ResponseInputContentUnionParam, error) {
	out := make([]responses.ResponseInputContentUnionParam, 0, len(items))

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
			out = append(out, responses.ResponseInputContentParamOfInputText(txt))

		case spec.ContentItemKindImage:
			if it.ImageItem == nil {
				continue
			}
			img := it.ImageItem

			var detail responses.ResponseInputImageDetail
			switch img.Detail {
			case spec.ImageDetailAuto:
				detail = responses.ResponseInputImageDetailAuto
			case spec.ImageDetailHigh:
				detail = responses.ResponseInputImageDetailHigh
			case spec.ImageDetailLow:
				detail = responses.ResponseInputImageDetailLow

			}

			// Prefer supplied data over URL.
			if data := strings.TrimSpace(img.ImageData); data != "" {
				mime := strings.TrimSpace(img.ImageMIME)
				if mime == "" {
					mime = spec.DefaultImageDataMIME
				}
				oaiImg := responses.ResponseInputImageParam{
					Detail:   detail,
					ImageURL: param.NewOpt(fmt.Sprintf("data:%s;base64,%s", mime, data)),
				}
				out = append(out, responses.ResponseInputContentUnionParam{
					OfInputImage: &oaiImg,
				})
			} else if u := strings.TrimSpace(img.ImageURL); u != "" {
				oaiImg := responses.ResponseInputImageParam{
					Detail:   detail,
					ImageURL: param.NewOpt(u),
				}
				out = append(out, responses.ResponseInputContentUnionParam{
					OfInputImage: &oaiImg,
				})
			} else {
				logutil.Debug("no data or url present for image", "id", img.ID, "name", img.ImageName)
			}

		case spec.ContentItemKindFile:
			if it.FileItem == nil {
				continue
			}
			f := it.FileItem

			// Prefer embedded data as true file input.
			if data := strings.TrimSpace(f.FileData); data != "" {
				mime := spec.DefaultFileDataMIME
				if f.FileMIME != "" {
					mime = f.FileMIME
				}
				dataURL := fmt.Sprintf("data:%s;base64,%s", mime, data)
				fileParam := responses.ResponseInputFileParam{
					FileData: param.NewOpt(dataURL),
					Filename: param.NewOpt(strings.TrimSpace(f.FileName)),
					Type:     openaiSharedConstant.InputFile("").Default(),
				}
				out = append(out, responses.ResponseInputContentUnionParam{
					OfInputFile: &fileParam,
				})
			} else if u := strings.TrimSpace(f.FileURL); u != "" {
				fileParam := responses.ResponseInputFileParam{
					FileURL: param.NewOpt(u),
					Type:    openaiSharedConstant.InputFile("").Default(),
				}
				out = append(out, responses.ResponseInputContentUnionParam{
					OfInputFile: &fileParam,
				})
			} else {
				logutil.Debug("no data or url present for file", "id", f.ID, "name", f.FileName)
			}
		case spec.ContentItemKindRefusal:
			// Refusal should not be present in InputMessage.
			continue
		default:
			logutil.Debug("unknown content for input messages", "kind", it.Kind)
		}
	}
	return out, nil
}

// contentItemsToOpenAI converts spec.Content items to OpenAI output message parts.
func contentItemsToOpenAIOutputContent(
	items []spec.InputOutputContentItemUnion,
) ([]responses.ResponseOutputMessageContentUnionParam, error) {
	out := make([]responses.ResponseOutputMessageContentUnionParam, 0, len(items))

	for _, it := range items {
		switch it.Kind {
		case spec.ContentItemKindText:
			if it.TextItem == nil {
				continue
			}
			annotations := citationsToAnnotations(it.TextItem.Citations)
			out = append(out, responses.ResponseOutputMessageContentUnionParam{
				OfOutputText: &responses.ResponseOutputTextParam{
					Annotations: annotations,
					Text:        it.TextItem.Text,
				},
			})

		case spec.ContentItemKindRefusal:
			if it.RefusalItem == nil {
				continue
			}
			out = append(out, responses.ResponseOutputMessageContentUnionParam{
				OfRefusal: &responses.ResponseOutputRefusalParam{
					Refusal: it.RefusalItem.Refusal,
				},
			})

		case spec.ContentItemKindImage, spec.ContentItemKindFile:
			// Image and PDF should not be present in OutputMessage.
		default:
			logutil.Debug("unknown content for output messages", "kind", it.Kind)
		}
	}
	return out, nil
}

func citationsToAnnotations(
	citations []spec.Citation,
) []responses.ResponseOutputTextAnnotationUnionParam {
	if len(citations) == 0 {
		return nil
	}
	out := make([]responses.ResponseOutputTextAnnotationUnionParam, 0)
	for _, a := range citations {
		// Only URL citations are currently supported.
		if a.URLCitation == nil {
			continue
		}
		out = append(out, responses.ResponseOutputTextAnnotationUnionParam{
			OfURLCitation: &responses.ResponseOutputTextAnnotationURLCitationParam{
				URL:        a.URLCitation.URL,
				Title:      a.URLCitation.Title,
				StartIndex: a.URLCitation.StartIndex,
				EndIndex:   a.URLCitation.EndIndex,
			},
		})
	}
	return out
}

// reasoningContentToOpenAIItem converts a generic ReasoningContent to an
// OpenAI Responses reasoning input item.
func reasoningContentToOpenAIItem(
	r *spec.ReasoningContent,
) *responses.ResponseInputItemUnionParam {
	if r == nil {
		return nil
	}

	status := responses.ResponseReasoningItemStatusCompleted
	if r.Status == fromOpenAIStatus(string(responses.ResponseReasoningItemStatusIncomplete)) {
		status = responses.ResponseReasoningItemStatusIncomplete
	} else if r.Status == fromOpenAIStatus(string(responses.ResponseReasoningItemStatusInProgress)) {
		status = responses.ResponseReasoningItemStatusInProgress
	}
	item := &responses.ResponseReasoningItemParam{
		ID:     r.ID,
		Status: status,
	}

	if len(r.EncryptedContent) > 0 {
		item.EncryptedContent = param.NewOpt(r.EncryptedContent[0])
	}

	if len(r.Summary) > 0 {
		item.Summary = make(
			[]responses.ResponseReasoningItemSummaryParam,
			0,
			len(r.Summary),
		)
		for _, s := range r.Summary {
			s = strings.TrimSpace(s)
			if s == "" {
				continue
			}
			item.Summary = append(item.Summary, responses.ResponseReasoningItemSummaryParam{
				Text: s,
			})
		}
	}

	if len(r.Thinking) > 0 {
		item.Content = make(
			[]responses.ResponseReasoningItemContentParam,
			0,
			len(r.Thinking),
		)
		for _, t := range r.Thinking {
			t = strings.TrimSpace(t)
			if t == "" {
				continue
			}
			item.Content = append(item.Content, responses.ResponseReasoningItemContentParam{
				Text: t,
			})
		}
	}

	return &responses.ResponseInputItemUnionParam{
		OfReasoning: item,
	}
}

func toolCallToOpenAIItem(call *spec.ToolCall) *responses.ResponseInputItemUnionParam {
	if call == nil || strings.TrimSpace(call.ID) == "" {
		return nil
	}
	switch call.Type {
	case spec.ToolTypeFunction:
		status := responses.ResponseFunctionToolCallStatusCompleted
		if call.Status == fromOpenAIStatus(string(responses.ResponseFunctionToolCallStatusIncomplete)) {
			status = responses.ResponseFunctionToolCallStatusIncomplete
		} else if call.Status == fromOpenAIStatus(string(responses.ResponseFunctionToolCallStatusInProgress)) {
			status = responses.ResponseFunctionToolCallStatusInProgress
		}

		fc := responses.ResponseFunctionToolCallParam{
			ID:        param.NewOpt(call.ID),
			CallID:    call.CallID,
			Name:      call.Name,
			Arguments: call.Arguments,
			Status:    status,
			Type:      openaiSharedConstant.FunctionCall("").Default(),
		}
		return &responses.ResponseInputItemUnionParam{
			OfFunctionCall: &fc,
		}

	case spec.ToolTypeCustom:
		cc := responses.ResponseCustomToolCallParam{
			ID:     param.NewOpt(call.ID),
			CallID: call.CallID,
			Name:   call.Name,
			Input:  call.Arguments,
			Type:   openaiSharedConstant.CustomToolCall("").Default(),
		}
		return &responses.ResponseInputItemUnionParam{
			OfCustomToolCall: &cc,
		}
	case spec.ToolTypeWebSearch:
		return webSearchToolCallToOpenAIResponses(call)

	default:
		return nil
	}
}

func webSearchToolCallToOpenAIResponses(
	toolCall *spec.ToolCall,
) *responses.ResponseInputItemUnionParam {
	if toolCall == nil || strings.TrimSpace(toolCall.ID) == "" || len(toolCall.WebSearchToolCallItems) == 0 {
		return nil
	}

	var status responses.ResponseFunctionWebSearchStatus

	switch toolCall.Status {
	case fromOpenAIStatus(string(responses.ResponseFunctionWebSearchStatusInProgress)):
		status = responses.ResponseFunctionWebSearchStatusInProgress
	case fromOpenAIStatus(string(responses.ResponseFunctionWebSearchStatusFailed)):
		status = responses.ResponseFunctionWebSearchStatusFailed
	case fromOpenAIStatus(string(responses.ResponseFunctionWebSearchStatusSearching)):
		status = responses.ResponseFunctionWebSearchStatusSearching
	default:
		status = responses.ResponseFunctionWebSearchStatusCompleted
	}

	out := &responses.ResponseInputItemUnionParam{
		OfWebSearchCall: &responses.ResponseFunctionWebSearchParam{
			ID:     toolCall.ID,
			Status: status,
			Type:   openaiSharedConstant.WebSearchCall("").Default(),
		},
	}
	// OpenAI has only 1 web search call item as of now.
	wcall := toolCall.WebSearchToolCallItems[0]

	action := &responses.ResponseFunctionWebSearchActionUnionParam{}
	switch wcall.Kind {
	case spec.WebSearchToolCallKindSearch:
		if wcall.SearchItem == nil {
			return nil
		}
		action.OfSearch = &responses.ResponseFunctionWebSearchActionSearchParam{
			Query:   wcall.SearchItem.Query,
			Sources: []responses.ResponseFunctionWebSearchActionSearchSourceParam{},
		}

		for _, u := range wcall.SearchItem.Sources {
			action.OfSearch.Sources = append(action.OfSearch.Sources,
				responses.ResponseFunctionWebSearchActionSearchSourceParam{
					URL: u.URL,
				},
			)
		}

	case spec.WebSearchToolCallKindOpenPage:
		if wcall.OpenPageItem == nil {
			return nil
		}

		action.OfOpenPage = &responses.ResponseFunctionWebSearchActionOpenPageParam{
			URL: wcall.OpenPageItem.URL,
		}

	case spec.WebSearchToolCallKindFind:
		if wcall.FindItem == nil {
			return nil
		}

		action.OfFind = &responses.ResponseFunctionWebSearchActionFindParam{
			Pattern: wcall.FindItem.Pattern,
			URL:     wcall.FindItem.URL,
		}

	default:
		return nil
	}

	if action.OfSearch != nil || action.OfOpenPage != nil || action.OfFind != nil {
		out.OfWebSearchCall.Action = *action
		return out
	}

	return nil
}

func toolOutputToOpenAIResponses(
	toolOutput *spec.ToolOutput,
) *responses.ResponseInputItemUnionParam {
	if toolOutput == nil || strings.TrimSpace(toolOutput.CallID) == "" {
		return nil
	}
	switch toolOutput.Type {
	case spec.ToolTypeFunction:
		status := responses.ResponseFunctionToolCallStatusCompleted
		if toolOutput.Status == fromOpenAIStatus(string(responses.ResponseFunctionToolCallStatusIncomplete)) {
			status = responses.ResponseFunctionToolCallStatusIncomplete
		} else if toolOutput.Status == fromOpenAIStatus(string(responses.ResponseFunctionToolCallStatusInProgress)) {
			status = responses.ResponseFunctionToolCallStatusInProgress
		}
		items, err := contentItemsToOpenAIFunctionCallOutputContent(toolOutput.Contents)
		if err != nil {
			return nil
		}
		if len(items) > 0 {
			return &responses.ResponseInputItemUnionParam{
				OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
					ID:     param.NewOpt(toolOutput.ID),
					CallID: toolOutput.CallID,
					Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
						OfResponseFunctionCallOutputItemArray: items,
					},
					Status: string(status),
					Type:   openaiSharedConstant.FunctionCallOutput("").Default(),
				},
			}
		}

	case spec.ToolTypeCustom:
		fcItems, err := contentItemsToOpenAIFunctionCallOutputContent(toolOutput.Contents)
		if err != nil {
			return nil
		}
		if len(fcItems) > 0 {
			items := make(
				[]responses.ResponseCustomToolCallOutputOutputOutputContentListItemUnionParam,
				0,
				len(fcItems),
			)
			for _, fi := range fcItems {
				i := &responses.ResponseCustomToolCallOutputOutputOutputContentListItemUnionParam{}
				if fi.OfInputText != nil {
					i.OfInputText = &responses.ResponseInputTextParam{
						Text: fi.OfInputText.Text,
					}
				}
				if fi.OfInputImage != nil {
					i.OfInputImage = &responses.ResponseInputImageParam{
						Detail:   responses.ResponseInputImageDetail(fi.OfInputImage.Detail),
						FileID:   fi.OfInputImage.FileID,
						ImageURL: fi.OfInputImage.ImageURL,
					}
				}
				if fi.OfInputFile != nil {
					i.OfInputFile = &responses.ResponseInputFileParam{
						FileID:   fi.OfInputFile.FileID,
						FileData: fi.OfInputFile.FileData,
						FileURL:  fi.OfInputFile.FileURL,
						Filename: fi.OfInputFile.Filename,
					}
				}
				if i.OfInputText != nil || i.OfInputImage != nil || i.OfInputFile != nil {
					items = append(items, *i)
				}
			}
			if len(items) > 0 {
				return &responses.ResponseInputItemUnionParam{
					OfCustomToolCallOutput: &responses.ResponseCustomToolCallOutputParam{
						ID:     param.NewOpt(toolOutput.ID),
						CallID: toolOutput.CallID,
						Output: responses.ResponseCustomToolCallOutputOutputUnionParam{
							OfOutputContentList: items,
						},
						Type: openaiSharedConstant.CustomToolCallOutput("").Default(),
					},
				}
			}
		}

	case spec.ToolTypeWebSearch:
		// OpenAI doesn't have web search tool output object.
	}
	return nil
}

// contentItemsToOpenAI converts spec.Content items to OpenAI input message parts.
func contentItemsToOpenAIFunctionCallOutputContent(
	items []spec.ToolOutputItemUnion,
) ([]responses.ResponseFunctionCallOutputItemUnionParam, error) {
	out := make([]responses.ResponseFunctionCallOutputItemUnionParam, 0, len(items))

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

			out = append(out, responses.ResponseFunctionCallOutputItemUnionParam{
				OfInputText: &responses.ResponseInputTextContentParam{
					Text: txt,
				},
			})

		case spec.ContentItemKindImage:
			if it.ImageItem == nil {
				continue
			}
			img := it.ImageItem

			var detail responses.ResponseInputImageContentDetail
			switch img.Detail {
			case spec.ImageDetailAuto:
				detail = responses.ResponseInputImageContentDetailAuto
			case spec.ImageDetailHigh:
				detail = responses.ResponseInputImageContentDetailHigh
			case spec.ImageDetailLow:
				detail = responses.ResponseInputImageContentDetailLow
			}

			// Prefer supplied data over URL.
			if data := strings.TrimSpace(img.ImageData); data != "" {
				mime := strings.TrimSpace(img.ImageMIME)
				if mime == "" {
					mime = spec.DefaultImageDataMIME
				}
				oaiImg := responses.ResponseInputImageContentParam{
					Detail:   detail,
					ImageURL: param.NewOpt(fmt.Sprintf("data:%s;base64,%s", mime, data)),
				}
				out = append(out, responses.ResponseFunctionCallOutputItemUnionParam{
					OfInputImage: &oaiImg,
				})
			} else if u := strings.TrimSpace(img.ImageURL); u != "" {
				oaiImg := responses.ResponseInputImageContentParam{
					Detail:   detail,
					ImageURL: param.NewOpt(u),
				}
				out = append(out, responses.ResponseFunctionCallOutputItemUnionParam{
					OfInputImage: &oaiImg,
				})
			} else {
				logutil.Debug("no data or url present for image", "id", img.ID, "name", img.ImageName)
			}

		case spec.ContentItemKindFile:
			if it.FileItem == nil {
				continue
			}
			f := it.FileItem

			// Prefer embedded data as true file input.
			if data := strings.TrimSpace(f.FileData); data != "" {
				mime := spec.DefaultFileDataMIME
				if f.FileMIME != "" {
					mime = f.FileMIME
				}
				dataURL := fmt.Sprintf("data:%s;base64,%s", mime, data)
				fileParam := responses.ResponseInputFileContentParam{
					FileData: param.NewOpt(dataURL),
					Filename: param.NewOpt(strings.TrimSpace(f.FileName)),
					Type:     openaiSharedConstant.InputFile("").Default(),
				}
				out = append(out, responses.ResponseFunctionCallOutputItemUnionParam{
					OfInputFile: &fileParam,
				})
			} else if u := strings.TrimSpace(f.FileURL); u != "" {
				fileParam := responses.ResponseInputFileContentParam{
					FileURL: param.NewOpt(u),
					Type:    openaiSharedConstant.InputFile("").Default(),
				}
				out = append(out, responses.ResponseFunctionCallOutputItemUnionParam{
					OfInputFile: &fileParam,
				})
			} else {
				logutil.Debug("no data or url present for file", "id", f.ID, "name", f.FileName)
			}
		case spec.ContentItemKindRefusal:
			// Refusal should not be present in call output.
			continue
		default:
			logutil.Debug("unknown content for input messages", "kind", it.Kind)
		}
	}
	return out, nil
}

func toolChoicesToOpenAIResponseTools(
	toolChoices []spec.ToolChoice,
) ([]responses.ToolUnionParam, map[string]spec.ToolChoice, error) {
	if len(toolChoices) == 0 {
		// Nothing to return.
		return []responses.ToolUnionParam{}, nil, nil
	}
	ordered, nameMap := sdkutil.BuildToolChoiceNameMapping(toolChoices)
	out := make([]responses.ToolUnionParam, 0, len(ordered))
	webSearchAdded := false

	for _, tw := range ordered {
		tc := tw.Choice
		name := tw.Name
		switch tc.Type {
		case spec.ToolTypeFunction, spec.ToolTypeCustom:
			if tc.Arguments == nil || name == "" {
				continue
			}
			// For now, both function and custom tools are expressed as function tools.
			fn := responses.FunctionToolParam{
				Name:        name,
				Parameters:  tc.Arguments,
				Type:        openaiSharedConstant.Function("function"),
				Description: param.NewOpt(sdkutil.ToolDescription(tc)),
			}

			out = append(out, responses.ToolUnionParam{OfFunction: &fn})

		case spec.ToolTypeWebSearch:
			if tc.WebSearchArguments == nil || webSearchAdded {
				// We add web search tool choice only once.
				continue
			}
			fn := responses.WebSearchToolParam{
				Type:              responses.WebSearchToolTypeWebSearch,
				SearchContextSize: responses.WebSearchToolSearchContextSizeMedium,
			}
			if len(tc.WebSearchArguments.AllowedDomains) != 0 {
				fn.Filters = responses.WebSearchToolFiltersParam{AllowedDomains: tc.WebSearchArguments.AllowedDomains}
			}
			if tc.WebSearchArguments.UserLocation != nil {
				fn.UserLocation = responses.WebSearchToolUserLocationParam{
					Type:     string(openaiSharedConstant.Approximate("").Default()),
					City:     param.NewOpt(tc.WebSearchArguments.UserLocation.City),
					Country:  param.NewOpt(tc.WebSearchArguments.UserLocation.Country),
					Region:   param.NewOpt(tc.WebSearchArguments.UserLocation.Region),
					Timezone: param.NewOpt(tc.WebSearchArguments.UserLocation.Timezone),
				}
			}
			switch tc.WebSearchArguments.SearchContextSize {
			case "low":
				fn.SearchContextSize = responses.WebSearchToolSearchContextSizeLow
			case "high":
				fn.SearchContextSize = responses.WebSearchToolSearchContextSizeHigh
			default:
				fn.SearchContextSize = responses.WebSearchToolSearchContextSizeMedium
			}

			out = append(out, responses.ToolUnionParam{OfWebSearch: &fn})
			webSearchAdded = true

		default:
			continue

		}
	}

	if len(out) == 0 {
		return []responses.ToolUnionParam{}, nil, nil
	}
	return out, nameMap, nil
}

func outputsFromOpenAIResponse(
	resp *responses.Response,
	toolChoiceNameMap map[string]spec.ToolChoice,
) []spec.OutputUnion {
	if resp == nil || len(resp.Output) == 0 {
		return nil
	}

	var outs []spec.OutputUnion

	for _, item := range resp.Output {
		switch item.Type {
		case string(openaiSharedConstant.Message("").Default()):
			m := item.AsMessage()

			// Treat as a single assistant message.
			outMsg := spec.InputOutputContent{
				ID:     m.ID,
				Role:   spec.RoleAssistant,
				Status: fromOpenAIStatus(string(m.Status)),
			}

			for _, c := range m.Content {
				// Text content with optional annotations -> ContentItemText.
				if txt := strings.TrimSpace(c.Text); txt != "" {
					textItem := spec.ContentItemText{
						Text:      c.Text,
						Citations: responsesAnnotationsToCitations(c.Annotations),
					}
					outMsg.Contents = append(
						outMsg.Contents,
						spec.InputOutputContentItemUnion{
							Kind:     spec.ContentItemKindText,
							TextItem: &textItem,
						},
					)
				}

				// Refusal (if present) -> ContentItemRefusal.
				if r := strings.TrimSpace(c.Refusal); r != "" {
					refItem := spec.ContentItemRefusal{
						Refusal: r,
					}
					outMsg.Contents = append(
						outMsg.Contents,
						spec.InputOutputContentItemUnion{
							Kind:        spec.ContentItemKindRefusal,
							RefusalItem: &refItem,
						},
					)
				}
			}

			if len(outMsg.Contents) > 0 {
				outs = append(
					outs,
					spec.OutputUnion{
						Kind:          spec.OutputKindOutputMessage,
						OutputMessage: &outMsg,
					},
				)
			}

		case string(openaiSharedConstant.Reasoning("").Default()):
			ri := item.AsReasoning()
			r := spec.ReasoningContent{
				ID:     ri.ID,
				Role:   spec.RoleAssistant,
				Status: fromOpenAIStatus(string(ri.Status)),
			}
			if ri.EncryptedContent != "" {
				r.EncryptedContent = []string{ri.EncryptedContent}
			}
			if len(ri.Summary) > 0 {
				for _, s := range ri.Summary {
					if txt := strings.TrimSpace(s.Text); txt != "" {
						r.Summary = append(r.Summary, txt)
					}
				}
			}
			if len(ri.Content) > 0 {
				for _, c := range ri.Content {
					if txt := strings.TrimSpace(c.Text); txt != "" {
						r.Thinking = append(r.Thinking, txt)
					}
				}
			}
			outs = append(
				outs,
				spec.OutputUnion{
					Kind:             spec.OutputKindReasoningMessage,
					ReasoningMessage: &r,
				},
			)

		case string(openaiSharedConstant.FunctionCall("").Default()):
			fn := item.AsFunctionCall()
			if fn.CallID == "" || strings.TrimSpace(fn.Name) == "" || toolChoiceNameMap == nil {
				continue
			}
			var choiceID string
			if choice, ok := toolChoiceNameMap[fn.Name]; ok {
				choiceID = choice.ID
			} else {
				continue
			}

			call := spec.ToolCall{
				ChoiceID:  choiceID,
				Type:      spec.ToolTypeFunction,
				Role:      spec.RoleAssistant,
				ID:        fn.ID,
				CallID:    fn.CallID,
				Name:      fn.Name,
				Arguments: fn.Arguments,
				Status:    fromOpenAIStatus(string(fn.Status)),
			}

			outs = append(
				outs,
				spec.OutputUnion{
					Kind:             spec.OutputKindFunctionToolCall,
					FunctionToolCall: &call,
				},
			)

		case string(openaiSharedConstant.CustomToolCall("").Default()):
			ct := item.AsCustomToolCall()
			if ct.CallID == "" || strings.TrimSpace(ct.Name) == "" || toolChoiceNameMap == nil {
				continue
			}

			var choiceID string
			if choice, ok := toolChoiceNameMap[ct.Name]; ok {
				choiceID = choice.ID
			} else {
				continue
			}

			call := spec.ToolCall{
				ChoiceID:  choiceID,
				Type:      spec.ToolTypeCustom,
				Role:      spec.RoleAssistant,
				ID:        ct.ID,
				CallID:    ct.CallID,
				Name:      ct.Name,
				Arguments: ct.Input,
				// No status to custom tool call. Consider completed.
				Status: spec.StatusCompleted,
			}

			outs = append(
				outs,
				spec.OutputUnion{
					Kind:           spec.OutputKindCustomToolCall,
					CustomToolCall: &call,
				},
			)
		case string(openaiSharedConstant.WebSearchCall("").Default()):
			ct := item.AsWebSearchCall()
			if ct.ID == "" || toolChoiceNameMap == nil {
				continue
			}
			// Web search calls don't carry a tool name in the API. We assume
			// there is at most one web_search ToolChoice and look it up by type.
			var choiceID string

			for _, choice := range toolChoiceNameMap {
				if choice.Type == spec.ToolTypeWebSearch {
					choiceID = choice.ID
					break
				}
			}

			if choiceID == "" {
				// No matching web_search ToolChoice; skip this call.
				continue
			}

			call := spec.ToolCall{
				ChoiceID:               choiceID,
				Type:                   spec.ToolTypeWebSearch,
				Role:                   spec.RoleAssistant,
				ID:                     ct.ID,
				CallID:                 ct.ID,
				Name:                   spec.DefaultWebSearchToolName,
				Status:                 fromOpenAIStatus(string(ct.Status)),
				WebSearchToolCallItems: make([]spec.WebSearchToolCallItemUnion, 0, 1),
			}
			webSearchItem := &spec.WebSearchToolCallItemUnion{}
			action := ct.Action
			switch action.Type {
			case "search":
				webSearchItem.Kind = spec.WebSearchToolCallKindSearch
				webSearchItem.SearchItem = &spec.WebSearchToolCallSearch{
					Query: action.Query,
				}
				sources := make([]spec.WebSearchToolCallSearchSource, 0, len(action.Sources))
				for _, s := range action.Sources {
					sources = append(sources, spec.WebSearchToolCallSearchSource{URL: s.URL})
				}
				webSearchItem.SearchItem.Sources = sources
				call.WebSearchToolCallItems = append(call.WebSearchToolCallItems, *webSearchItem)
			case "open_page":
				webSearchItem.Kind = spec.WebSearchToolCallKindOpenPage
				webSearchItem.OpenPageItem = &spec.WebSearchToolCallOpenPage{
					URL: action.URL,
				}

				call.WebSearchToolCallItems = append(call.WebSearchToolCallItems, *webSearchItem)
			case "find":
				webSearchItem.Kind = spec.WebSearchToolCallKindFind
				webSearchItem.FindItem = &spec.WebSearchToolCallFind{
					URL:     action.URL,
					Pattern: action.Pattern,
				}
				call.WebSearchToolCallItems = append(call.WebSearchToolCallItems, *webSearchItem)
			}
			outs = append(
				outs,
				spec.OutputUnion{
					Kind:              spec.OutputKindWebSearchToolCall,
					WebSearchToolCall: &call,
				},
			)
		}
	}

	return outs
}

func responsesAnnotationsToCitations(
	anns []responses.ResponseOutputTextAnnotationUnion,
) []spec.Citation {
	if len(anns) == 0 {
		return nil
	}
	out := make([]spec.Citation, 0)
	for _, a := range anns {
		if a.Type != string(openaiSharedConstant.URLCitation("").Default()) {
			// Only URL citations are currently supported.
			continue
		}
		out = append(out, spec.Citation{
			Kind: spec.CitationKindURL,
			URLCitation: &spec.URLCitation{
				URL:        a.URL,
				Title:      a.Title,
				StartIndex: a.StartIndex,
				EndIndex:   a.EndIndex,
			},
		})
	}
	return out
}

// usageFromOpenAIResponse normalizes OpenAI Responses API usage into spec.Usage.
func usageFromOpenAIResponse(resp *responses.Response) *spec.Usage {
	uOut := &spec.Usage{}
	if resp == nil {
		return uOut
	}

	u := resp.Usage

	uOut.InputTokensTotal = u.InputTokens
	uOut.InputTokensCached = u.InputTokensDetails.CachedTokens
	uOut.InputTokensUncached = max(u.InputTokens-u.InputTokensDetails.CachedTokens, 0)
	uOut.OutputTokens = u.OutputTokens
	uOut.ReasoningTokens = u.OutputTokensDetails.ReasoningTokens

	return uOut
}

func toOpenAIStatus(status spec.Status) string {
	switch status {
	case spec.StatusInProgress:
		return "in_progress"
	default:
		return string(status)
	}
}

func fromOpenAIStatus(status string) spec.Status {
	switch status {
	case "in_progress":
		return spec.StatusInProgress
	default:
		return spec.Status(status)
	}
}
