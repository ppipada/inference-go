package anthropicsdk

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"maps"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	anthropicSharedConstant "github.com/anthropics/anthropic-sdk-go/shared/constant"

	"github.com/ppipada/inference-go/internal/debugclient"
	"github.com/ppipada/inference-go/internal/sdkutil"
	"github.com/ppipada/inference-go/spec"
)

// AnthropicMessagesAPI implements CompletionProvider for Anthropics' Messages API.
type AnthropicMessagesAPI struct {
	ProviderParam *spec.ProviderParam
	Debug         bool
	client        *anthropic.Client
}

// NewAnthropicMessagesAPI creates a new instance of Anthropics provider.
func NewAnthropicMessagesAPI(
	pi spec.ProviderParam,
	debug bool,
) (*AnthropicMessagesAPI, error) {
	if pi.Name == "" || pi.Origin == "" {
		return nil, errors.New("anthropic messages api LLM: invalid args")
	}
	return &AnthropicMessagesAPI{
		ProviderParam: &pi,
		Debug:         debug,
	}, nil
}

func (api *AnthropicMessagesAPI) InitLLM(ctx context.Context) error {
	if !api.IsConfigured(ctx) {
		slog.Debug(
			string(
				api.ProviderParam.Name,
			) + ": No API key given. Not initializing Anthropics client",
		)
		return nil
	}

	opts := []option.RequestOption{
		// Sets x-api-key.
		option.WithAPIKey(api.ProviderParam.APIKey),
	}

	providerURL := spec.DefaultAnthropicOrigin
	if api.ProviderParam.Origin != "" {
		baseURL := strings.TrimSuffix(api.ProviderParam.Origin, "/")
		// Remove 'v1/messages' from pathPrefix if present,
		// This is because anthropic sdk adds 'v1/messages' internally.
		pathPrefix := strings.TrimSuffix(
			api.ProviderParam.ChatCompletionPathPrefix,
			"v1/messages",
		)
		providerURL = baseURL + pathPrefix
		opts = append(opts, option.WithBaseURL(strings.TrimSuffix(providerURL, "/")))
	}

	// Add default headers.
	for k, v := range api.ProviderParam.DefaultHeaders {
		opts = append(opts, option.WithHeader(strings.TrimSpace(k), strings.TrimSpace(v)))
	}

	// If the caller provided a non-standard API key header, attach it.
	if api.ProviderParam.APIKeyHeaderKey != "" &&
		!strings.EqualFold(
			api.ProviderParam.APIKeyHeaderKey,
			spec.DefaultAnthropicAuthorizationHeaderKey,
		) &&
		!strings.EqualFold(
			api.ProviderParam.APIKeyHeaderKey,
			spec.DefaultAuthorizationHeaderKey,
		) {
		opts = append(
			opts,
			option.WithHeader(api.ProviderParam.APIKeyHeaderKey, api.ProviderParam.APIKey),
		)
	}

	dbgCfg := debugclient.DefaultDebugConfig
	dbgCfg.LogToSlog = api.Debug
	httpClient := debugclient.NewDebugHTTPClient(dbgCfg)
	opts = append(opts, option.WithHTTPClient(httpClient))

	c := anthropic.NewClient(opts...)
	api.client = &c
	slog.Info(
		"anthropic messages api LLM provider initialized",
		"name", string(api.ProviderParam.Name),
		"URL", providerURL,
	)
	return nil
}

func (api *AnthropicMessagesAPI) DeInitLLM(ctx context.Context) error {
	api.client = nil
	slog.Info(
		"anthropic messages api LLM: provider de initialized",
		"name",
		string(api.ProviderParam.Name),
	)
	return nil
}

func (api *AnthropicMessagesAPI) GetProviderInfo(ctx context.Context) *spec.ProviderParam {
	return api.ProviderParam
}

func (api *AnthropicMessagesAPI) IsConfigured(ctx context.Context) bool {
	return api.ProviderParam != nil && api.ProviderParam.APIKey != ""
}

func (api *AnthropicMessagesAPI) SetProviderAPIKey(ctx context.Context, apiKey string) error {
	if apiKey == "" {
		return errors.New("anthropic messages api LLM: invalid apikey provided")
	}
	if api.ProviderParam == nil {
		return errors.New("anthropic messages api LLM: no ProviderParam found")
	}
	api.ProviderParam.APIKey = apiKey
	return nil
}

func (api *AnthropicMessagesAPI) FetchCompletion(
	ctx context.Context,
	req *spec.FetchCompletionRequest,
	onStreamTextData func(textData string) error,
	onStreamThinkingData func(thinkingData string) error,
) (*spec.FetchCompletionResponse, error) {
	if api.client == nil {
		return nil, errors.New("anthropic messages api LLM: client not initialized")
	}
	if req == nil || len(req.Inputs) == 0 || req.ModelParam.Name == "" {
		return nil, errors.New("anthropic messages api LLM: empty completion data")
	}

	// Build Anthropic input messages + system blocks.
	msgs, sysParams, err := toAnthropicMessagesInput(
		ctx,
		req.ModelParam.SystemPrompt,
		req.Inputs,
	)
	if err != nil {
		return nil, err
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(req.ModelParam.Name),
		MaxTokens: int64(req.ModelParam.MaxOutputLength),
		Messages:  msgs,
	}
	if len(sysParams) > 0 {
		params.System = sysParams
	}

	if rp := req.ModelParam.Reasoning; rp != nil {
		switch rp.Type {
		case spec.ReasoningTypeHybridWithTokens:
			// Use the explicit token budget, enforcing a minimum if provided.
			if rp.Tokens > 0 {
				budget := max(rp.Tokens, 1024)
				params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(budget))
			}

		case spec.ReasoningTypeSingleWithLevels:
			// Map qualitative levels to a default token budget; ignore rp.Tokens.
			var budget int
			switch rp.Level {
			case spec.ReasoningLevelNone:
				// No reasoning.
			case spec.ReasoningLevelMinimal, spec.ReasoningLevelLow:
				budget = 1024
			case spec.ReasoningLevelMedium:
				budget = 2048
			case spec.ReasoningLevelHigh:
				budget = 8192
			case spec.ReasoningLevelXHigh:
				budget = 16384
			default:
				// Unknown level -> leave Thinking unset.
			}
			if budget > 0 {
				params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(budget))
			}
		}
	}

	if t := req.ModelParam.Temperature; t != nil {
		params.Temperature = anthropic.Float(*t)
	}

	timeout := spec.DefaultAPITimeout
	if req.ModelParam.Timeout > 0 {
		timeout = time.Duration(req.ModelParam.Timeout) * time.Second
	}

	var toolChoiceNameMap map[string]spec.ToolChoice
	if len(req.ToolChoices) > 0 {
		toolDefs, nameMap, err := toolChoicesToAnthropicTools(req.ToolChoices)
		if err != nil {
			return nil, err
		}
		if len(toolDefs) > 0 {
			params.Tools = toolDefs
			toolChoiceNameMap = nameMap
		}
	}

	ctx = debugclient.AddDebugResponseToCtx(ctx)

	if req.ModelParam.Stream && onStreamTextData != nil && onStreamThinkingData != nil {
		return api.doStreaming(ctx, params, onStreamTextData, onStreamThinkingData, timeout, toolChoiceNameMap)
	}
	return api.doNonStreaming(ctx, params, timeout, toolChoiceNameMap)
}

func (api *AnthropicMessagesAPI) doNonStreaming(
	ctx context.Context,
	params anthropic.MessageNewParams,
	timeout time.Duration,
	toolChoiceNameMap map[string]spec.ToolChoice,
) (*spec.FetchCompletionResponse, error) {
	resp := &spec.FetchCompletionResponse{}

	msg, err := api.client.Messages.New(ctx, params, option.WithRequestTimeout(timeout))
	isNilResp := msg == nil || len(msg.Content) == 0
	sdkutil.AttachDebugResp(ctx, resp, err, isNilResp, msg)
	resp.Usage = usageFromAnthropicMessage(msg)
	if err != nil {
		resp.Error = &spec.Error{Message: err.Error()}
		return resp, err
	}
	if !isNilResp {
		resp.Outputs = outputsFromAnthropicMessage(msg, toolChoiceNameMap)
	}

	return resp, nil
}

func (api *AnthropicMessagesAPI) doStreaming(
	ctx context.Context,
	params anthropic.MessageNewParams,
	onStreamTextData, onStreamThinkingData func(string) error,
	timeout time.Duration,
	toolChoiceNameMap map[string]spec.ToolChoice,
) (*spec.FetchCompletionResponse, error) {
	resp := &spec.FetchCompletionResponse{}
	writeTextData, flushTextData := sdkutil.NewBufferedStreamer(
		onStreamTextData,
		sdkutil.FlushInterval,
		sdkutil.FlushChunkSize,
	)
	writeThinkingData, flushThinkingData := sdkutil.NewBufferedStreamer(
		onStreamThinkingData,
		sdkutil.FlushInterval,
		sdkutil.FlushChunkSize,
	)

	stream := api.client.Messages.NewStreaming(
		ctx,
		params,
		option.WithRequestTimeout(timeout),
	)
	defer func() { _ = stream.Close() }()

	var (
		respFull            anthropic.Message
		streamWriteErr      error
		streamAccumulateErr error
	)

	for stream.Next() {
		event := stream.Current()
		err := respFull.Accumulate(event)
		if err != nil {
			streamAccumulateErr = err
			break
		}

		switch eventVariant := event.AsAny().(type) {
		case anthropic.MessageStartEvent:
			// Contains a Message object with empty content (metadata only).
		case anthropic.MessageDeltaEvent:
			// Top-level message metadata changes; nothing to stream to user.
		case anthropic.MessageStopEvent:
			// Conversation turn complete.
		case anthropic.ContentBlockStopEvent:
			// Content block done.
		case anthropic.ContentBlockStartEvent:
			streamWriteErr = handleContentBlockStartEvent(eventVariant, writeTextData, writeThinkingData)
			if streamWriteErr != nil {
				break
			}
		case anthropic.ContentBlockDeltaEvent:
			streamWriteErr = handleContentBlockDeltaEvent(eventVariant, writeTextData, writeThinkingData)
			if streamWriteErr != nil {
				break
			}
		default:
			// No valid variant.
		}
		// If downstream write failed (client disconnect, etc.), stop consuming the stream.
		if streamWriteErr != nil {
			break
		}
	}

	if flushTextData != nil {
		flushTextData()
	}

	if flushThinkingData != nil {
		flushThinkingData()
	}

	streamErr := errors.Join(stream.Err(), streamAccumulateErr, streamWriteErr)
	isNilResp := len(respFull.Content) == 0
	sdkutil.AttachDebugResp(ctx, resp, streamErr, isNilResp, &respFull)
	resp.Usage = usageFromAnthropicMessage(&respFull)
	if streamErr != nil {
		resp.Error = &spec.Error{Message: streamErr.Error()}
	}

	if !isNilResp {
		resp.Outputs = outputsFromAnthropicMessage(&respFull, toolChoiceNameMap)
	}

	return resp, streamErr
}

func handleContentBlockStartEvent(
	event anthropic.ContentBlockStartEvent,
	writeTextData, writeThinkingData func(string) error,
) error {
	switch cb := event.ContentBlock.AsAny().(type) {
	case anthropic.TextBlock:
		return writeTextData(cb.Text)

	case anthropic.ThinkingBlock:
		return writeThinkingData(cb.Thinking)

	case anthropic.RedactedThinkingBlock:
		// We don't stream redacted thinking to the caller.
	case anthropic.ToolUseBlock:
	case anthropic.ServerToolUseBlock:
	case anthropic.WebSearchToolResultBlock:
	default:
		// Unknown or future content block type.
	}
	return nil
}

func handleContentBlockDeltaEvent(
	event anthropic.ContentBlockDeltaEvent,
	writeTextData, writeThinkingData func(string) error,
) error {
	switch delta := event.Delta.AsAny().(type) {
	case anthropic.TextDelta:
		return writeTextData(delta.Text)

	case anthropic.ThinkingDelta:
		return writeThinkingData(delta.Thinking)

	case anthropic.InputJSONDelta:
	case anthropic.CitationsDelta:
	case anthropic.SignatureDelta:
	default:
		// Unknown or future delta variant.
	}
	return nil
}

// toAnthropicMessagesInput converts a sequence of generic InputUnion items into
// Anthropic MessageParam and system prompt blocks.
func toAnthropicMessagesInput(
	_ context.Context,
	systemPrompt string,
	inputs []spec.InputUnion,
) (msgs []anthropic.MessageParam, sysPrompts []anthropic.TextBlockParam, err error) {
	var out []anthropic.MessageParam
	var sysParts []string

	if s := strings.TrimSpace(systemPrompt); s != "" {
		sysParts = append(sysParts, s)
	}

	for _, in := range inputs {
		if sdkutil.IsInputUnionEmpty(in) {
			continue
		}

		switch in.Kind {
		case spec.InputKindInputMessage:
			// User messages only.
			if in.InputMessage == nil || in.InputMessage.Role != spec.RoleUser {
				continue
			}
			blocks := contentItemsToAnthropicContentBlocks(in.InputMessage.Contents)
			if len(blocks) == 0 {
				continue
			}
			out = append(out, anthropic.NewUserMessage(blocks...))

		case spec.InputKindOutputMessage:
			// Assistant messages (prior turns).
			if in.OutputMessage == nil || in.OutputMessage.Role != spec.RoleAssistant {
				continue
			}
			blocks := contentItemsToAnthropicContentBlocks(in.OutputMessage.Contents)
			if len(blocks) == 0 {
				continue
			}
			out = append(out, anthropic.NewAssistantMessage(blocks...))

		case spec.InputKindReasoningMessage:
			if in.ReasoningMessage == nil {
				continue
			}
			block := reasoningContentToAnthropicBlocks(in.ReasoningMessage)
			if block != nil {
				out = append(out, anthropic.NewAssistantMessage(*block))
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

			block := toolCallToAnthropicToolUseBlock(call)
			if block != nil {
				out = append(out, anthropic.NewAssistantMessage(*block))
			}

		case spec.InputKindFunctionToolOutput, spec.InputKindCustomToolOutput, spec.InputKindWebSearchToolOutput:
			isWebSearchOutput := false
			var output *spec.ToolOutput
			switch {
			case in.FunctionToolOutput != nil:
				output = in.FunctionToolOutput
			case in.CustomToolOutput != nil:
				output = in.CustomToolOutput
			case in.WebSearchToolOutput != nil:
				output = in.WebSearchToolOutput
				isWebSearchOutput = true
			}
			block := toolOutputToAnthropicBlocks(output)
			if block != nil {
				if isWebSearchOutput {
					out = append(out, anthropic.NewAssistantMessage(*block))
				} else {
					out = append(out, anthropic.NewUserMessage(*block))
				}
			}

		default:
			// Unknown input kind.
		}
	}

	// System prompt as a single text block.
	if len(sysParts) > 0 {
		sysStr := strings.Join(sysParts, "\n\n")
		sysPrompts = append(sysPrompts, anthropic.TextBlockParam{Text: sysStr})
	}

	return out, sysPrompts, nil
}

// contentItemsToAnthropicContentBlocks converts generic content items into Anthropic
// content blocks (text/image/document).
func contentItemsToAnthropicContentBlocks(
	items []spec.InputOutputContentItemUnion,
) []anthropic.ContentBlockParamUnion {
	if len(items) == 0 {
		return nil
	}
	out := make([]anthropic.ContentBlockParamUnion, 0, len(items))

	for _, it := range items {
		switch it.Kind {
		case spec.ContentItemKindText:
			tb := contentItemTextToAnthropicTextBlockParam(it.TextItem)
			if tb != nil {
				out = append(out, anthropic.ContentBlockParamUnion{OfText: tb})
			}

		case spec.ContentItemKindImage:
			ib := contentItemImageToAnthropicImageBlockParam(it.ImageItem)
			if ib != nil {
				out = append(out, anthropic.ContentBlockParamUnion{OfImage: ib})
			}

		case spec.ContentItemKindFile:
			db := contentItemFileToAnthropicDocumentBlockParam(it.FileItem)
			if db != nil {
				out = append(out, anthropic.ContentBlockParamUnion{OfDocument: db})
			}

		case spec.ContentItemKindRefusal:
			// Anthropic does not have a dedicated "refusal" content block type.
			// Refusals are conveyed via stop_reason="refusal". We don't send
			// refusals back as input content.
			continue

		default:
			slog.Debug("anthropic: unknown content item kind for message", "kind", it.Kind)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func reasoningContentToAnthropicBlocks(
	r *spec.ReasoningContent,
) *anthropic.ContentBlockParamUnion {
	if r == nil {
		return nil
	}

	if len(r.RedactedThinking) > 0 {
		// If redacted thinking is present it is redacted thinking block.
		data := strings.Join(r.RedactedThinking, " ")
		out := anthropic.NewRedactedThinkingBlock(data)
		return &out
	}

	if len(r.Thinking) > 0 && r.Signature != "" {
		data := strings.Join(r.Thinking, " ")
		out := anthropic.NewThinkingBlock(r.Signature, data)
		return &out
	}
	return nil
}

// toolCallToAnthropicToolUseBlock converts a ToolCall into an Anthropic tool_use block.
func toolCallToAnthropicToolUseBlock(
	toolCall *spec.ToolCall,
) *anthropic.ContentBlockParamUnion {
	if toolCall == nil || strings.TrimSpace(toolCall.ID) == "" {
		return nil
	}
	switch toolCall.Type {
	case spec.ToolTypeFunction, spec.ToolTypeCustom:
		if strings.TrimSpace(toolCall.Name) == "" {
			return nil
		}
		args := strings.TrimSpace(toolCall.Arguments)
		if args == "" {
			args = "{}"
		}
		raw := json.RawMessage(args)

		return &anthropic.ContentBlockParamUnion{OfToolUse: &anthropic.ToolUseBlockParam{
			ID:    toolCall.ID,
			Name:  toolCall.Name,
			Input: raw,
		}}

	case spec.ToolTypeWebSearch:
		if len(toolCall.WebSearchToolCallItems) == 0 {
			return nil
		}
		// Anthropic has only 1 web search call item as of now.
		wcall := toolCall.WebSearchToolCallItems[0]
		if wcall.Kind != spec.WebSearchToolCallKindSearch || wcall.SearchItem == nil || wcall.SearchItem.Input == nil {
			// Only search supported.
			return nil
		}

		return &anthropic.ContentBlockParamUnion{OfServerToolUse: &anthropic.ServerToolUseBlockParam{
			ID:    toolCall.ID,
			Input: wcall.SearchItem.Input,
			Name:  anthropicSharedConstant.WebSearch("").Default(),
		}}

	}
	return nil
}

func toolOutputToAnthropicBlocks(
	toolOutput *spec.ToolOutput,
) *anthropic.ContentBlockParamUnion {
	if toolOutput == nil || strings.TrimSpace(toolOutput.CallID) == "" {
		return nil
	}

	switch toolOutput.Type {
	case spec.ToolTypeFunction, spec.ToolTypeCustom:
		items := contentItemsToAnthropicToolResultBlocks(toolOutput.Contents)
		if len(items) == 0 {
			return nil
		}
		toolBlock := anthropic.ToolResultBlockParam{
			ToolUseID: toolOutput.CallID,
			Content:   items,
			IsError:   anthropic.Bool(toolOutput.IsError),
		}
		return &anthropic.ContentBlockParamUnion{OfToolResult: &toolBlock}

	case spec.ToolTypeWebSearch:
		content := webSearchToolOutputItemsToAnthropicWebSearchContent(
			toolOutput.WebSearchToolOutputItems,
		)
		if content == nil {
			return nil
		}
		wsBlock := anthropic.WebSearchToolResultBlockParam{
			ToolUseID: toolOutput.CallID,
			Content:   *content,
			// CacheControl omitted; add mapping from toolOutput.CacheControl if needed.
			// Type omitted; zero value marshals as "web_search_tool_result".
		}
		return &anthropic.ContentBlockParamUnion{OfWebSearchToolResult: &wsBlock}
	default:
		// Nothing to do more.
	}
	return nil
}

func contentItemsToAnthropicToolResultBlocks(
	items []spec.ToolOutputItemUnion,
) []anthropic.ToolResultBlockParamContentUnion {
	if len(items) == 0 {
		return nil
	}
	out := make([]anthropic.ToolResultBlockParamContentUnion, 0, len(items))

	for _, it := range items {
		switch it.Kind {
		case spec.ContentItemKindText:
			tb := contentItemTextToAnthropicTextBlockParam(it.TextItem)
			if tb != nil {
				out = append(out, anthropic.ToolResultBlockParamContentUnion{OfText: tb})
			}

		case spec.ContentItemKindImage:
			ib := contentItemImageToAnthropicImageBlockParam(it.ImageItem)
			if ib != nil {
				out = append(out, anthropic.ToolResultBlockParamContentUnion{OfImage: ib})
			}

		case spec.ContentItemKindFile:
			db := contentItemFileToAnthropicDocumentBlockParam(it.FileItem)
			if db != nil {
				out = append(out, anthropic.ToolResultBlockParamContentUnion{OfDocument: db})
			}
		case spec.ContentItemKindRefusal:
			// Invalid for this.
		default:
			slog.Debug("anthropic: unknown content item kind for message", "kind", it.Kind)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func webSearchToolOutputItemsToAnthropicWebSearchContent(
	items []spec.WebSearchToolOutputItemUnion,
) *anthropic.WebSearchToolResultBlockParamContentUnion {
	if len(items) == 0 {
		return nil
	}

	// If there's an error item, treat the whole tool call as an error.
	for _, it := range items {
		if it.Kind == spec.WebSearchToolOutputKindError && it.ErrorItem != nil {
			errParam := anthropic.WebSearchToolRequestErrorParam{
				// Code is something like "invalid_tool_input", "unavailable", etc.
				ErrorCode: anthropic.WebSearchToolRequestErrorErrorCode(it.ErrorItem.Code),
				// Type is omitted; zero value marshals as "web_search_tool_result_error".
			}
			return &anthropic.WebSearchToolResultBlockParamContentUnion{
				OfRequestWebSearchToolResultError: &errParam,
			}
		}
	}

	// Otherwise, collect all search results.
	results := make([]anthropic.WebSearchResultBlockParam, 0, len(items))

	for _, it := range items {
		if it.Kind != spec.WebSearchToolOutputKindSearch || it.SearchItem == nil {
			continue
		}

		ws := it.SearchItem

		block := anthropic.WebSearchResultBlockParam{
			URL:              ws.URL,
			Title:            ws.Title,
			EncryptedContent: ws.EncryptedContent,
			// Type omitted; zero value marshals as "web_search_result".
		}

		// Optional page_age.
		if s := strings.TrimSpace(ws.PageAge); s != "" {
			block.PageAge = anthropic.String(s)
		}

		results = append(results, block)
	}

	if len(results) == 0 {
		return nil
	}

	return &anthropic.WebSearchToolResultBlockParamContentUnion{
		OfWebSearchToolResultBlockItem: results,
	}
}

func contentItemTextToAnthropicTextBlockParam(textItem *spec.ContentItemText) *anthropic.TextBlockParam {
	if textItem == nil {
		return nil
	}
	text := strings.TrimSpace(textItem.Text)
	if text == "" {
		return nil
	}
	tb := &anthropic.TextBlockParam{
		Text: text,
	}

	if anns := citationsToAnthropicTextCitations(textItem.Citations); len(anns) > 0 {
		tb.Citations = anns
	}
	return tb
}

// citationsToAnthropicTextCitations converts our generic URL citations into
// Anthropic TextCitationParamUnion (web_search_result_location).
func citationsToAnthropicTextCitations(
	citations []spec.Citation,
) []anthropic.TextCitationParamUnion {
	if len(citations) == 0 {
		return nil
	}
	out := make([]anthropic.TextCitationParamUnion, 0)
	for _, c := range citations {
		if c.URLCitation == nil {
			continue
		}
		out = append(out, anthropic.TextCitationParamUnion{
			OfWebSearchResultLocation: &anthropic.CitationWebSearchResultLocationParam{
				CitedText:      c.URLCitation.CitedText,
				EncryptedIndex: c.URLCitation.EncryptedIndex,
				Title:          anthropic.String(c.URLCitation.Title),
				URL:            c.URLCitation.URL,
			},
		})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func contentItemImageToAnthropicImageBlockParam(imageItem *spec.ContentItemImage) *anthropic.ImageBlockParam {
	if imageItem == nil {
		return nil
	}

	if data := strings.TrimSpace(imageItem.ImageData); data != "" {
		mime := strings.TrimSpace(imageItem.ImageMIME)
		if mime == "" {
			mime = spec.DefaultImageDataMIME
		}
		return &anthropic.ImageBlockParam{
			Source: anthropic.ImageBlockParamSourceUnion{
				OfBase64: &anthropic.Base64ImageSourceParam{
					Data:      data,
					MediaType: anthropic.Base64ImageSourceMediaType(mime),
				},
			},
		}
	} else if u := strings.TrimSpace(imageItem.ImageURL); u != "" {
		return &anthropic.ImageBlockParam{
			Source: anthropic.ImageBlockParamSourceUnion{
				OfURL: &anthropic.URLImageSourceParam{
					URL: u,
				},
			},
		}
	}
	return nil
}

func contentItemFileToAnthropicDocumentBlockParam(fileItem *spec.ContentItemFile) *anthropic.DocumentBlockParam {
	if fileItem == nil {
		return nil
	}
	data := strings.TrimSpace(fileItem.FileData)
	url := strings.TrimSpace(fileItem.FileURL)
	mime := strings.TrimSpace(fileItem.FileMIME)
	// Map files to document blocks where possible.
	switch {
	case data != "" && strings.HasPrefix(mime, "application/pdf"):
		return &anthropic.DocumentBlockParam{
			Source: anthropic.DocumentBlockParamSourceUnion{
				OfBase64: &anthropic.Base64PDFSourceParam{
					Data: data,
				},
			},
		}

	case url != "" && strings.HasPrefix(mime, "application/pdf"):
		return &anthropic.DocumentBlockParam{
			Source: anthropic.DocumentBlockParamSourceUnion{
				OfURL: &anthropic.URLPDFSourceParam{
					URL: url,
				},
			},
		}

	case data != "" && strings.HasPrefix(mime, "text/"):
		// For plain text, Anthropic expects actual text, not base64. If you
		// want to support this fully, decode base64 here. For now we skip.
		slog.Debug("anthropic: skipping non-pdf base64 file; plain-text decoding not implemented",
			"id", fileItem.ID, "name", fileItem.FileName, "mime", mime)
	default:
		// Other file types not supported as document blocks.
	}
	return nil
}

func toolChoicesToAnthropicTools(
	toolChoices []spec.ToolChoice,
) ([]anthropic.ToolUnionParam, map[string]spec.ToolChoice, error) {
	if len(toolChoices) == 0 {
		return []anthropic.ToolUnionParam{}, nil, nil
	}

	ordered, nameMap := sdkutil.BuildToolChoiceNameMapping(toolChoices)
	out := make([]anthropic.ToolUnionParam, 0, len(ordered))
	webSearchAdded := false

	for _, tw := range ordered {
		tc := tw.Choice
		name := tw.Name
		switch tc.Type {
		case spec.ToolTypeFunction, spec.ToolTypeCustom:
			if name == "" || tc.Arguments == nil {
				continue
			}

			// Copy schema so we can safely manipulate.
			schema := make(map[string]any, len(tc.Arguments))
			maps.Copy(schema, tc.Arguments)

			inputSchema := anthropic.ToolInputSchemaParam{
				Type: anthropicSharedConstant.Object("object"),
			}
			if tVal, ok := schema["type"].(string); ok && strings.TrimSpace(tVal) != "" {
				inputSchema.Type = anthropicSharedConstant.Object(strings.ToLower(strings.TrimSpace(tVal)))
				delete(schema, "type")
			}
			if props, ok := schema["properties"]; ok {
				inputSchema.Properties = props
				delete(schema, "properties")
			}
			if req, ok := schema["required"]; ok {
				switch v := req.(type) {
				case []any:
					required := make([]string, 0, len(v))
					for _, item := range v {
						if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
							required = append(required, strings.TrimSpace(s))
						}
					}
					if len(required) > 0 {
						inputSchema.Required = required
					}
				case []string:
					required := make([]string, 0, len(v))
					for _, item := range v {
						if strings.TrimSpace(item) != "" {
							required = append(required, strings.TrimSpace(item))
						}
					}
					if len(required) > 0 {
						inputSchema.Required = required
					}
				}
				delete(schema, "required")
			}
			if len(schema) > 0 {
				inputSchema.ExtraFields = schema
			}

			toolUnion := anthropic.ToolUnionParamOfTool(inputSchema, name)
			if variant := toolUnion.OfTool; variant != nil {
				if desc := sdkutil.ToolDescription(tc); desc != "" {
					variant.Description = anthropic.String(desc)
				}
			}
			out = append(out, toolUnion)

		case spec.ToolTypeWebSearch:
			if tc.WebSearchArguments == nil || webSearchAdded {
				// We add web search tool choice only once.
				continue
			}
			ws := tc.WebSearchArguments

			wsTool := anthropic.WebSearchTool20250305Param{}

			if len(ws.AllowedDomains) > 0 && len(ws.BlockedDomains) > 0 {
				slog.Warn(
					"anthropic: web_search tool has both allowed_domains and blocked_domains; using allowed_domains only",
					"toolID",
					tc.ID,
				)
			}
			if len(ws.AllowedDomains) > 0 {
				wsTool.AllowedDomains = ws.AllowedDomains
			} else if len(ws.BlockedDomains) > 0 {
				wsTool.BlockedDomains = ws.BlockedDomains
			}
			if ws.MaxUses > 0 {
				wsTool.MaxUses = anthropic.Int(ws.MaxUses)
			}
			if ws.UserLocation != nil {
				wsTool.UserLocation = anthropic.WebSearchTool20250305UserLocationParam{
					City:     anthropic.String(ws.UserLocation.City),
					Country:  anthropic.String(ws.UserLocation.Country),
					Region:   anthropic.String(ws.UserLocation.Region),
					Timezone: anthropic.String(ws.UserLocation.Timezone),
				}
			}

			out = append(out, anthropic.ToolUnionParam{
				OfWebSearchTool20250305: &wsTool,
			})
			webSearchAdded = true
		}

	}

	if len(out) == 0 {
		return []anthropic.ToolUnionParam{}, nil, nil
	}
	return out, nameMap, nil
}

func outputsFromAnthropicMessage(
	msg *anthropic.Message,
	toolChoiceNameMap map[string]spec.ToolChoice,
) []spec.OutputUnion {
	if msg == nil || len(msg.Content) == 0 {
		return nil
	}

	var outs []spec.OutputUnion

	msgStatus := mapAnthropicStopReasonToStatus(msg.StopReason)

	// We emit function/custom tool calls and web search server tool uses
	// immediately as separate OutputUnion entries when we encounter them.
	for _, content := range msg.Content {
		switch v := content.AsAny().(type) {
		case anthropic.TextBlock:
			assistantMsg := spec.InputOutputContent{
				ID:   msg.ID,
				Role: spec.RoleAssistant,
				// Anthropic doesn't expose per-block status; use stop_reason.
				Status: msgStatus,
			}
			txt := strings.TrimSpace(v.Text)
			if txt == "" {
				continue
			}
			textItem := spec.ContentItemText{
				Text:      v.Text,
				Citations: anthropicCitationsToSpec(v.Citations),
			}
			assistantMsg.Contents = []spec.InputOutputContentItemUnion{{
				Kind:     spec.ContentItemKindText,
				TextItem: &textItem,
			}}

			outs = append(
				outs,
				spec.OutputUnion{
					Kind:          spec.OutputKindOutputMessage,
					OutputMessage: &assistantMsg,
				},
			)

		case anthropic.ThinkingBlock:
			r := spec.ReasoningContent{
				ID:        msg.ID,
				Role:      spec.RoleAssistant,
				Status:    msgStatus,
				Signature: v.Signature,
				Thinking:  []string{v.Thinking},
			}
			outs = append(
				outs,
				spec.OutputUnion{
					Kind:             spec.OutputKindReasoningMessage,
					ReasoningMessage: &r,
				},
			)

		case anthropic.RedactedThinkingBlock:
			r := spec.ReasoningContent{
				ID:               msg.ID,
				Role:             spec.RoleAssistant,
				Status:           msgStatus,
				RedactedThinking: []string{v.Data},
			}
			outs = append(
				outs,
				spec.OutputUnion{
					Kind:             spec.OutputKindReasoningMessage,
					ReasoningMessage: &r,
				},
			)

		case anthropic.ToolUseBlock:
			// Client tool call.
			name := strings.TrimSpace(v.Name)
			id := strings.TrimSpace(v.ID)
			if id == "" || name == "" {
				continue
			}

			var (
				choiceID string
				toolType = spec.ToolTypeFunction
			)
			if toolChoiceNameMap != nil {
				if tc, ok := toolChoiceNameMap[name]; ok {
					choiceID = tc.ID
					toolType = tc.Type
				}
			}
			if choiceID == "" {
				continue
			}

			call := spec.ToolCall{
				ChoiceID: choiceID,
				Type:     toolType,
				Role:     spec.RoleAssistant,
				ID:       id,
				CallID:   id,
				Name:     v.Name,
				Arguments: strings.TrimSpace(
					string(v.Input),
				),
				Status: spec.StatusCompleted,
			}

			var kind spec.OutputKind
			switch toolType {
			case spec.ToolTypeFunction:
				kind = spec.OutputKindFunctionToolCall
			case spec.ToolTypeCustom:
				kind = spec.OutputKindCustomToolCall
			default:
				kind = spec.OutputKindFunctionToolCall
			}

			out := spec.OutputUnion{Kind: kind}
			switch kind {
			case spec.OutputKindCustomToolCall:
				out.CustomToolCall = &call
			case spec.OutputKindFunctionToolCall:
				out.FunctionToolCall = &call
			default:
			}
			outs = append(outs, out)

		case anthropic.ServerToolUseBlock:
			// Anthropic server web search tool call.
			id := strings.TrimSpace(v.ID)
			if id == "" {
				continue
			}

			var choiceID string

			for _, tc := range toolChoiceNameMap {
				if tc.Type == spec.ToolTypeWebSearch {
					choiceID = tc.ID
					break
				}
			}

			if choiceID == "" {
				continue
			}

			call := spec.ToolCall{
				ChoiceID:               choiceID,
				Type:                   spec.ToolTypeWebSearch,
				Role:                   spec.RoleAssistant,
				ID:                     id,
				CallID:                 id,
				Name:                   spec.DefaultWebSearchToolName,
				Status:                 spec.StatusCompleted,
				WebSearchToolCallItems: anthropicServerToolInputToWebSearchCallItems(v.Input),
			}

			outs = append(
				outs,
				spec.OutputUnion{
					Kind:              spec.OutputKindWebSearchToolCall,
					WebSearchToolCall: &call,
				},
			)

		case anthropic.WebSearchToolResultBlock:
			// Map the result back to the web_search ToolChoice, if any.
			var choiceID string

			for _, tc := range toolChoiceNameMap {
				if tc.Type == spec.ToolTypeWebSearch {
					choiceID = tc.ID
					break
				}
			}

			wsOut := &spec.ToolOutput{
				ChoiceID: choiceID,
				Type:     spec.ToolTypeWebSearch,
				Role:     spec.RoleAssistant,
				ID:       v.ToolUseID,
				CallID:   v.ToolUseID,
				Status:   spec.StatusCompleted,
				Name:     spec.DefaultWebSearchToolName,
			}

			if v.Content.ErrorCode != "" {
				wsOut.IsError = true
				wsOut.WebSearchToolOutputItems = []spec.WebSearchToolOutputItemUnion{
					{
						Kind: spec.WebSearchToolOutputKindError,
						ErrorItem: &spec.WebSearchToolOutputError{
							Code: string(v.Content.ErrorCode),
						},
					},
				}
				outs = append(
					outs,
					spec.OutputUnion{
						Kind:                spec.OutputKindWebSearchToolOutput,
						WebSearchToolOutput: wsOut,
					},
				)
			} else {
				if len(v.Content.OfWebSearchResultBlockArray) == 0 {
					continue
				}
				wsOut.WebSearchToolOutputItems = make([]spec.WebSearchToolOutputItemUnion, 0, len(v.Content.OfWebSearchResultBlockArray))
				for _, w := range v.Content.OfWebSearchResultBlockArray {
					search := &spec.WebSearchToolOutputSearch{
						URL:              w.URL,
						Title:            w.Title,
						EncryptedContent: w.EncryptedContent,
						PageAge:          w.PageAge,
					}
					wsOut.WebSearchToolOutputItems = append(wsOut.WebSearchToolOutputItems, spec.WebSearchToolOutputItemUnion{
						Kind:       spec.WebSearchToolOutputKindSearch,
						SearchItem: search,
					})

				}
				outs = append(
					outs,
					spec.OutputUnion{
						Kind:                spec.OutputKindWebSearchToolOutput,
						WebSearchToolOutput: wsOut,
					},
				)
			}
		default:
			// Future content variants.
		}
	}

	return outs
}

// anthropicCitationsToSpec converts Anthropic text citations into generic URL
// citations (only web_search_result_location is currently supported).
func anthropicCitationsToSpec(
	anns []anthropic.TextCitationUnion,
) []spec.Citation {
	if len(anns) == 0 {
		return nil
	}
	out := make([]spec.Citation, 0)
	for _, cc := range anns {
		if cc.Type != string(anthropicSharedConstant.WebSearchResultLocation("").Default()) {
			continue
		}
		out = append(out, spec.Citation{
			Kind: spec.CitationKindURL,
			URLCitation: &spec.URLCitation{
				URL:            cc.URL,
				Title:          cc.Title,
				CitedText:      cc.CitedText,
				EncryptedIndex: cc.EncryptedIndex,
			},
		})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// anthropicServerToolInputToWebSearchCallItems converts the server web search
// input payload into our generic WebSearchToolCall items.
func anthropicServerToolInputToWebSearchCallItems(
	input any,
) []spec.WebSearchToolCallItemUnion {
	if input == nil {
		return nil
	}

	if m, ok := input.(map[string]any); ok {

		item := spec.WebSearchToolCallItemUnion{
			Kind: spec.WebSearchToolCallKindSearch,
			SearchItem: &spec.WebSearchToolCallSearch{
				Input: m,
			},
		}
		if q, ok := m["query"].(string); ok {
			item.SearchItem.Query = q
		}
		return []spec.WebSearchToolCallItemUnion{item}

	}
	return nil
}

func mapAnthropicStopReasonToStatus(stopReason anthropic.StopReason) spec.Status {
	switch stopReason {
	case anthropic.StopReasonMaxTokens:
		return spec.StatusIncomplete
	case anthropic.StopReasonRefusal, anthropic.StopReasonPauseTurn, anthropic.StopReasonStopSequence:
		return spec.StatusFailed
	case anthropic.StopReasonEndTurn, anthropic.StopReasonToolUse:
		return spec.StatusCompleted
	}
	return spec.StatusCompleted
}

// usageFromAnthropicMessage normalizes Anthropic usage into spec.Usage.
func usageFromAnthropicMessage(msg *anthropic.Message) *spec.Usage {
	uOut := &spec.Usage{}
	if msg == nil {
		return uOut
	}

	u := msg.Usage

	uOut.InputTokensCached = u.CacheReadInputTokens
	uOut.InputTokensUncached = u.InputTokens
	uOut.InputTokensTotal = u.CacheReadInputTokens + u.InputTokens
	uOut.OutputTokens = u.OutputTokens
	// Anthropic does not currently expose explicit reasoning token counts.
	uOut.ReasoningTokens = 0

	return uOut
}
