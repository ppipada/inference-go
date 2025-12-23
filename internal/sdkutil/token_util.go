package sdkutil

import (
	"regexp"
	"strings"

	"github.com/ppipada/inference-go/internal/logutil"
	"github.com/ppipada/inference-go/spec"
)

func FilterMessagesByTokenCount(
	messages []spec.InputUnion,
	maxTokenCount int,
) []spec.InputUnion {
	if len(messages) == 0 {
		return nil
	}

	totalTokens := 0
	var filtered []spec.InputUnion

	// 1) Basic token-based filtering, newest-first.
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		tokensInMsg := countHeuristicTokensInInputUnion(msg)

		if totalTokens+tokensInMsg <= maxTokenCount || len(filtered) == 0 {
			filtered = append(filtered, msg)
			totalTokens += tokensInMsg

			if totalTokens > maxTokenCount {
				break
			}
		} else {
			break
		}
	}

	// 2) Reverse back to chronological order.
	for i, j := 0, len(filtered)-1; i < j; i, j = i+1, j-1 {
		filtered[i], filtered[j] = filtered[j], filtered[i]
	}

	// 3) Prune orphan tool outputs (those whose CallID has no matching ToolCall).
	filtered = pruneOrphanToolOutputs(filtered)

	if len(filtered) < len(messages) {
		logutil.Debug(
			"filtered messages are less than input",
			"originalCount", len(messages),
			"filteredCount", len(filtered),
			"approxTokens", totalTokens,
		)
	}

	return filtered
}

func pruneOrphanToolOutputs(msgs []spec.InputUnion) []spec.InputUnion {
	if len(msgs) == 0 {
		return msgs
	}

	// First collect all CallIDs for ToolCalls that are kept.
	callIDs := make(map[string]struct{})

	for _, in := range msgs {
		var call *spec.ToolCall

		switch in.Kind {
		case spec.InputKindFunctionToolCall:
			call = in.FunctionToolCall
		case spec.InputKindCustomToolCall:
			call = in.CustomToolCall
		case spec.InputKindWebSearchToolCall:
			call = in.WebSearchToolCall
		default:
			continue
		}

		if call == nil {
			continue
		}

		callID := strings.TrimSpace(call.CallID)
		if callID == "" {
			continue
		}

		callIDs[callID] = struct{}{}
	}

	// Then drop any ToolOutput/WebSearchToolOutput without a matching CallID.
	out := make([]spec.InputUnion, 0, len(msgs))
	for _, in := range msgs {
		var toolOut *spec.ToolOutput

		switch in.Kind {
		case spec.InputKindFunctionToolOutput:
			toolOut = in.FunctionToolOutput
		case spec.InputKindCustomToolOutput:
			toolOut = in.CustomToolOutput
		case spec.InputKindWebSearchToolOutput:
			toolOut = in.WebSearchToolOutput
		default:
		}

		if toolOut != nil {
			callID := strings.TrimSpace(toolOut.CallID)
			if callID != "" {
				if _, ok := callIDs[callID]; !ok {
					// Orphan output => drop it.
					continue
				}
			}
		}

		out = append(out, in)
	}

	return out
}

func countHeuristicTokensInInputUnion(in spec.InputUnion) int {
	switch in.Kind {
	case spec.InputKindInputMessage:
		return countTokensInInputOutputContent(in.InputMessage)

	case spec.InputKindOutputMessage:
		return countTokensInInputOutputContent(in.OutputMessage)

	case spec.InputKindReasoningMessage:
		return countTokensInReasoningContent(in.ReasoningMessage)

	case spec.InputKindFunctionToolCall:
		return countTokensInToolCall(in.FunctionToolCall)

	case spec.InputKindCustomToolCall:
		return countTokensInToolCall(in.CustomToolCall)

	case spec.InputKindWebSearchToolCall:
		return countTokensInToolCall(in.WebSearchToolCall)

	case spec.InputKindFunctionToolOutput:
		return countTokensInToolOutput(in.FunctionToolOutput)

	case spec.InputKindCustomToolOutput:
		return countTokensInToolOutput(in.CustomToolOutput)

	case spec.InputKindWebSearchToolOutput:
		return countTokensInToolOutput(in.WebSearchToolOutput)

	default:
		return 0
	}
}

func countTokensInInputOutputContent(c *spec.InputOutputContent) int {
	if c == nil {
		return 0
	}
	total := 0
	for _, it := range c.Contents {
		switch it.Kind {
		case spec.ContentItemKindText:
			if it.TextItem != nil {
				total += countHeuristicTokensInString(it.TextItem.Text)
			}
		case spec.ContentItemKindRefusal:
			if it.RefusalItem != nil {
				total += countHeuristicTokensInString(it.RefusalItem.Refusal)
			}
		case spec.ContentItemKindImage:
			// Ignore.
		case spec.ContentItemKindFile:
			if it.FileItem != nil {
				// AdditionalContext is the main textual part.
				total += countHeuristicTokensInString(it.FileItem.AdditionalContext)
			}
		}
	}
	return total
}

func countTokensInReasoningContent(r *spec.ReasoningContent) int {
	if r == nil {
		return 0
	}
	total := 0
	for _, s := range r.Summary {
		total += countHeuristicTokensInString(s)
	}
	for _, t := range r.Thinking {
		total += countHeuristicTokensInString(t)
	}
	for _, t := range r.RedactedThinking {
		total += countHeuristicTokensInString(t)
	}
	// EncryptedContent is opaque; ignore for heuristic token counting.
	return total
}

func countTokensInToolCall(call *spec.ToolCall) int {
	if call == nil {
		return 0
	}
	total := 0

	// Tool name + raw arguments text.
	total += countHeuristicTokensInString(call.Name)
	total += countHeuristicTokensInString(call.Arguments)

	// For web search calls, queries and patterns matter most.
	for _, item := range call.WebSearchToolCallItems {
		switch item.Kind {
		case spec.WebSearchToolCallKindSearch:
			if item.SearchItem != nil {
				total += countHeuristicTokensInString(item.SearchItem.Query)
			}
		case spec.WebSearchToolCallKindFind:
			if item.FindItem != nil {
				total += countHeuristicTokensInString(item.FindItem.Pattern)
			}
		case spec.WebSearchToolCallKindOpenPage:
			// URL only; typically short. Ignored for simplicity.
		}
	}

	return total
}

func countTokensInToolOutput(out *spec.ToolOutput) int {
	if out == nil {
		return 0
	}
	total := 0

	// Function/custom outputs: text content items.
	for _, it := range out.Contents {
		if it.Kind == spec.ContentItemKindText && it.TextItem != nil {
			total += countHeuristicTokensInString(it.TextItem.Text)
		}
	}

	// Web search outputs: titles + rendered content carry most of the text.
	for _, it := range out.WebSearchToolOutputItems {
		if it.Kind == spec.WebSearchToolOutputKindSearch && it.SearchItem != nil {
			total += countHeuristicTokensInString(it.SearchItem.Title)
			total += countHeuristicTokensInString(it.SearchItem.RenderedContent)
		}
		// Error items are usually tiny; we ignore them.
	}

	return total
}

var tokenRegex = regexp.MustCompile(`\w+|[^\s\w]`)

// countHeuristicTokensInString approximates token count by splitting into
// word-like chunks and single punctuation/symbol characters. This tends
// to be closer to modern OpenAI BPE tokenization than splitting only on
// whitespace.
func countHeuristicTokensInString(content string) int {
	content = strings.TrimSpace(content)
	if content == "" {
		return 0
	}

	matches := tokenRegex.FindAllString(content, -1)
	return len(matches)
}
