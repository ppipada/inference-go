package openairesponsessdk

import (
	"strings"

	"github.com/flexigpt/inference-go/internal/logutil"
	"github.com/flexigpt/inference-go/internal/sdkutil"
	"github.com/flexigpt/inference-go/spec"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

// reasoningContentToOpenAIItem converts a generic ReasoningContent to an
// OpenAI Responses reasoning input item.
func reasoningContentToOpenAIItem(
	r *spec.ReasoningContent,
) *responses.ResponseInputItemUnionParam {
	if r == nil {
		return nil
	}

	var status responses.ResponseReasoningItemStatus

	switch r.Status {
	case fromOpenAIStatus(string(responses.ResponseReasoningItemStatusCompleted)):
		status = responses.ResponseReasoningItemStatusCompleted
	case fromOpenAIStatus(string(responses.ResponseReasoningItemStatusIncomplete)):
		status = responses.ResponseReasoningItemStatusIncomplete
	case fromOpenAIStatus(string(responses.ResponseReasoningItemStatusInProgress)):
		status = responses.ResponseReasoningItemStatusInProgress
	default:

	}

	item := &responses.ResponseReasoningItemParam{
		ID:     r.ID,
		Status: status,
	}

	if enc, ok := firstNonEmptyEncrypted(r.EncryptedContent); ok {
		item.EncryptedContent = param.NewOpt(enc)
	}

	item.Summary = make([]responses.ResponseReasoningItemSummaryParam, 0)
	if len(r.Summary) > 0 {
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

// sanitizeReasoningInputs enforces the policy for OpenAI Responses:
//   - If any reasoning message contains encrypted_content => keep ONLY those reasoning messages,
//     and strip them down to encrypted_content only.
//   - If no reasoning message contains encrypted_content => drop ALL reasoning messages (fail-safe).
//
// This prevents leaking or incorrectly forwarding signature-based / plaintext reasoning content
// (e.g. from other providers) into the OpenAI Responses API.
func sanitizeReasoningInputs(inputs []spec.InputUnion) []spec.InputUnion {
	if len(inputs) == 0 {
		return nil
	}

	hasEncrypted := false
	for _, in := range inputs {
		if in.Kind != spec.InputKindReasoningMessage || sdkutil.IsInputUnionEmpty(in) || in.ReasoningMessage == nil {
			continue
		}
		if _, ok := firstNonEmptyEncrypted(in.ReasoningMessage.EncryptedContent); ok {
			hasEncrypted = true
			break
		}
	}

	out := make([]spec.InputUnion, 0, len(inputs))
	droppedReasoning := 0
	keptReasoning := 0

	for _, in := range inputs {
		if in.Kind != spec.InputKindReasoningMessage {
			out = append(out, in)
			continue
		}

		// Reasoning message sanitization.
		if sdkutil.IsInputUnionEmpty(in) || in.ReasoningMessage == nil {
			droppedReasoning++
			continue
		}

		enc, ok := firstNonEmptyEncrypted(in.ReasoningMessage.EncryptedContent)
		if !hasEncrypted {
			// No encrypted reasoning anywhere => drop all reasoning messages (fail-safe).
			droppedReasoning++
			continue
		}
		if !ok {
			// Mixed signature/plaintext + encrypted => keep encrypted only.
			droppedReasoning++
			continue
		}

		// Keep encrypted_content only.
		rc := *in.ReasoningMessage // copy
		rc.Signature = ""
		rc.Summary = nil
		rc.Thinking = nil
		rc.RedactedThinking = nil
		rc.EncryptedContent = []string{enc}

		inCopy := in
		inCopy.ReasoningMessage = &rc
		out = append(out, inCopy)
		keptReasoning++
	}

	if droppedReasoning > 0 {
		logutil.Debug(
			"openai responses: sanitized reasoning messages",
			"hasEncrypted", hasEncrypted,
			"kept", keptReasoning,
			"dropped", droppedReasoning,
		)
	}

	return out
}

func firstNonEmptyEncrypted(items []string) (string, bool) {
	for _, s := range items {
		if v := strings.TrimSpace(s); v != "" {
			return v, true
		}
	}
	return "", false
}
