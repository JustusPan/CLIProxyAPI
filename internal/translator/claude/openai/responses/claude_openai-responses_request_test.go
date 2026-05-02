package responses

import (
	"encoding/base64"
	"strings"
	"testing"

	sigcompat "github.com/router-for-me/CLIProxyAPI/v7/internal/signature"
	"github.com/tidwall/gjson"
	"google.golang.org/protobuf/encoding/protowire"
)

func TestConvertOpenAIResponsesRequestToClaude_SanitizesToolCallIDsForClaude(t *testing.T) {
	inputJSON := `{
		"model": "gpt-4.1",
		"input": [
			{
				"type": "function_call",
				"call_id": "call.with space:1",
				"name": "Read",
				"arguments": "{\"path\":\"README.md\"}"
			},
			{
				"type": "function_call_output",
				"call_id": "call.with space:1",
				"output": "ok"
			}
		]
	}`

	result := ConvertOpenAIResponsesRequestToClaude("claude-sonnet-4-5", []byte(inputJSON), false)
	resultJSON := gjson.ParseBytes(result)
	toolUseID := resultJSON.Get("messages.0.content.0.id").String()
	toolResultID := resultJSON.Get("messages.1.content.0.tool_use_id").String()

	if toolUseID != "call_with_space_1" {
		t.Fatalf("tool_use id = %q, want %q", toolUseID, "call_with_space_1")
	}
	if toolResultID != toolUseID {
		t.Fatalf("tool_result tool_use_id = %q, want same sanitized id %q", toolResultID, toolUseID)
	}
}

func TestConvertOpenAIResponsesRequestToClaude_ReasoningItemToThinkingBlock(t *testing.T) {
	rawSignature, expectedSignature := testClaudeResponsesThinkingSignature(t)
	raw := []byte(`{
		"model":"claude-test",
		"input":[
			{
				"type":"reasoning",
				"encrypted_content":"` + rawSignature + `",
				"summary":[{"type":"summary_text","text":"internal reasoning"}]
			},
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"visible answer"}]
			},
			{
				"type":"message",
				"role":"user",
				"content":[{"type":"input_text","text":"continue"}]
			}
		]
	}`)

	out := ConvertOpenAIResponsesRequestToClaude("claude-test", raw, false)
	root := gjson.ParseBytes(out)

	assistant := root.Get("messages.0")
	if got := assistant.Get("role").String(); got != "assistant" {
		t.Fatalf("first message role = %q, want assistant. Output: %s", got, string(out))
	}
	if got := assistant.Get("content.0.type").String(); got != "thinking" {
		t.Fatalf("first content type = %q, want thinking. Output: %s", got, string(out))
	}
	if got := assistant.Get("content.0.signature").String(); got != expectedSignature {
		t.Fatalf("thinking signature = %q, want %q", got, expectedSignature)
	}
	if got := assistant.Get("content.0.thinking").String(); got != "internal reasoning" {
		t.Fatalf("thinking text = %q, want internal reasoning", got)
	}
	if got := assistant.Get("content.1.type").String(); got != "text" {
		t.Fatalf("second content type = %q, want text. Output: %s", got, string(out))
	}
	if got := assistant.Get("content.1.text").String(); got != "visible answer" {
		t.Fatalf("assistant text = %q, want visible answer", got)
	}
	if got := root.Get("messages.1.role").String(); got != "user" {
		t.Fatalf("second message role = %q, want user. Output: %s", got, string(out))
	}
}

func TestConvertOpenAIResponsesRequestToClaude_SignatureOnlyReasoningFlushesBeforeUser(t *testing.T) {
	rawSignature, expectedSignature := testClaudeResponsesThinkingSignature(t)
	raw := []byte(`{
		"model":"claude-test",
		"input":[
			{
				"type":"reasoning",
				"encrypted_content":"` + rawSignature + `",
				"summary":[]
			},
			{
				"type":"message",
				"role":"user",
				"content":[{"type":"input_text","text":"continue"}]
			}
		]
	}`)

	out := ConvertOpenAIResponsesRequestToClaude("claude-test", raw, false)
	root := gjson.ParseBytes(out)

	thinking := root.Get("messages.0.content.0")
	if got := thinking.Get("type").String(); got != "thinking" {
		t.Fatalf("first content type = %q, want thinking. Output: %s", got, string(out))
	}
	if got := thinking.Get("signature").String(); got != expectedSignature {
		t.Fatalf("thinking signature = %q, want %q", got, expectedSignature)
	}
	if got := thinking.Get("thinking").String(); got != "" {
		t.Fatalf("thinking text = %q, want empty", got)
	}
	if got := root.Get("messages.1.role").String(); got != "user" {
		t.Fatalf("second message role = %q, want user. Output: %s", got, string(out))
	}
}

func TestConvertOpenAIResponsesRequestToClaude_DropsIncompatibleReasoningSignature(t *testing.T) {
	raw := []byte(`{
		"model":"claude-test",
		"input":[
			{
				"type":"reasoning",
				"encrypted_content":"` + testGPTResponsesReasoningSignature() + `",
				"summary":[{"type":"summary_text","text":"must not become Claude thinking"}]
			},
			{
				"type":"message",
				"role":"user",
				"content":[{"type":"input_text","text":"continue"}]
			}
		]
	}`)

	out := ConvertOpenAIResponsesRequestToClaude("claude-test", raw, false)

	if gjson.GetBytes(out, "messages.0.content.0.type").String() == "thinking" {
		t.Fatalf("GPT encrypted_content should not become Claude thinking. Output: %s", string(out))
	}
	if gjson.GetBytes(out, "messages.0.content.0.signature").Exists() {
		t.Fatalf("incompatible signature should not be forwarded. Output: %s", string(out))
	}
	if got := gjson.GetBytes(out, "messages.0.role").String(); got != "user" {
		t.Fatalf("first message role = %q, want user. Output: %s", got, string(out))
	}
}

func TestConvertOpenAIResponsesRequestToClaude_FunctionCallOutputPreservesInputImage(t *testing.T) {
	const imageB64 = "iVBORw0KGgo="
	dataURL := "data:image/png;base64," + imageB64
	raw := []byte(`{
		"model":"claude-test",
		"input":[
			{
				"type":"function_call",
				"call_id":"call_view_image_1",
				"name":"view_image",
				"arguments":"{}"
			},
			{
				"type":"function_call_output",
				"call_id":"call_view_image_1",
				"output":[
					{
						"type":"input_image",
						"image_url":"` + dataURL + `",
						"detail":"high"
					}
				]
			}
		]
	}`)

	out := ConvertOpenAIResponsesRequestToClaude("claude-test", raw, false)
	root := gjson.ParseBytes(out)

	toolResult := root.Get("messages.1.content.0")
	if got := toolResult.Get("type").String(); got != "tool_result" {
		t.Fatalf("tool_result type = %q, want tool_result. Output: %s", got, string(out))
	}
	if got := toolResult.Get("content.0.type").String(); got != "image" {
		t.Fatalf("tool_result content block type = %q, want image. Output: %s", got, string(out))
	}
	if got := toolResult.Get("content.0.source.media_type").String(); got != "image/png" {
		t.Fatalf("image media_type = %q, want image/png. Output: %s", got, string(out))
	}
	if got := toolResult.Get("content.0.source.data").String(); got != imageB64 {
		t.Fatalf("image data = %q, want raw base64 without data URL prefix", got)
	}
	if strings.Contains(toolResult.Get("content").Raw, "data:image") {
		t.Fatalf("tool_result content must not embed data URL as text. Output: %s", string(out))
	}
}

func TestConvertOpenAIResponsesRequestToClaude_KeepsToolUseAdjacentToToolResult(t *testing.T) {
	raw := []byte(`{
		"model":"claude-test",
		"input":[
			{
				"type":"function_call",
				"call_id":"call_00_awGuheXs4aRbtedNK8LE3743",
				"name":"js",
				"arguments":"{\"code\":\"nodeRepl.write('ok')\",\"title\":\"List Obsidian vault contents\"}"
			},
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"I'll check your Obsidian vault for articles."}]
			},
			{
				"type":"function_call_output",
				"call_id":"call_00_awGuheXs4aRbtedNK8LE3743",
				"output":"Wall time: 0.1963 seconds\nOutput:\n[{\"type\":\"text\",\"text\":\"\"}]"
			}
		]
	}`)

	out := ConvertOpenAIResponsesRequestToClaude("claude-test", raw, false)
	root := gjson.ParseBytes(out)

	if got := root.Get("messages.0.role").String(); got != "assistant" {
		t.Fatalf("first message role = %q, want assistant. Output: %s", got, string(out))
	}
	if got := root.Get("messages.0.content").String(); got != "I'll check your Obsidian vault for articles." {
		t.Fatalf("first message content = %q, want assistant text. Output: %s", got, string(out))
	}
	if got := root.Get("messages.1.content.0.type").String(); got != "tool_use" {
		t.Fatalf("second message first content type = %q, want tool_use. Output: %s", got, string(out))
	}
	if got := root.Get("messages.1.content.0.id").String(); got != "call_00_awGuheXs4aRbtedNK8LE3743" {
		t.Fatalf("tool_use id = %q, want call_00_awGuheXs4aRbtedNK8LE3743. Output: %s", got, string(out))
	}
	if got := root.Get("messages.2.content.0.type").String(); got != "tool_result" {
		t.Fatalf("third message first content type = %q, want tool_result. Output: %s", got, string(out))
	}
	if got := root.Get("messages.2.content.0.tool_use_id").String(); got != "call_00_awGuheXs4aRbtedNK8LE3743" {
		t.Fatalf("tool_result id = %q, want call_00_awGuheXs4aRbtedNK8LE3743. Output: %s", got, string(out))
	}
}

func TestConvertOpenAIResponsesRequestToClaude_DropsApplyPatchCustomTool(t *testing.T) {
	raw := []byte(`{
		"model":"claude-test",
		"input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"tools":[
			{
				"type":"custom",
				"name":"apply_patch",
				"description":"Use the apply_patch tool to edit files.",
				"format":{"type":"grammar","syntax":"lark","definition":"start: patch"}
			},
			{
				"type":"function",
				"name":"exec_command",
				"description":"Runs a command.",
				"parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}
			}
		]
	}`)

	out := ConvertOpenAIResponsesRequestToClaude("claude-test", raw, false)
	root := gjson.ParseBytes(out)

	if got := root.Get("tools.#").Int(); got != 1 {
		t.Fatalf("tools count = %d, want 1. Output: %s", got, string(out))
	}
	if got := root.Get("tools.0.name").String(); got != "exec_command" {
		t.Fatalf("tools.0.name = %q, want exec_command. Output: %s", got, string(out))
	}
	if got := root.Get("tools.#(name==\"apply_patch\")").Raw; got != "" {
		t.Fatalf("apply_patch custom tool should be dropped. Output: %s", string(out))
	}
}

func testClaudeResponsesThinkingSignature(t *testing.T) (string, string) {
	t.Helper()
	channelBlock := []byte{}
	channelBlock = protowire.AppendTag(channelBlock, 1, protowire.VarintType)
	channelBlock = protowire.AppendVarint(channelBlock, 12)
	channelBlock = protowire.AppendTag(channelBlock, 2, protowire.VarintType)
	channelBlock = protowire.AppendVarint(channelBlock, 2)
	channelBlock = protowire.AppendTag(channelBlock, 6, protowire.BytesType)
	channelBlock = protowire.AppendString(channelBlock, "claude-sonnet-4-6")

	container := []byte{}
	container = protowire.AppendTag(container, 1, protowire.BytesType)
	container = protowire.AppendBytes(container, channelBlock)

	payload := []byte{}
	payload = protowire.AppendTag(payload, 2, protowire.BytesType)
	payload = protowire.AppendBytes(payload, container)
	payload = protowire.AppendTag(payload, 3, protowire.VarintType)
	payload = protowire.AppendVarint(payload, 1)

	rawSignature := base64.StdEncoding.EncodeToString(payload)
	normalized, ok := sigcompat.CompatibleSignatureForProvider(sigcompat.SignatureProviderClaude, rawSignature)
	if !ok {
		t.Fatal("test Claude signature should be compatible")
	}
	return rawSignature, normalized
}

func testGPTResponsesReasoningSignature() string {
	payload := make([]byte, 1+8+16+16+32)
	payload[0] = 0x80
	payload[8] = 1
	for i := 9; i < len(payload); i++ {
		payload[i] = byte(i)
	}
	return base64.URLEncoding.EncodeToString(payload)
}

func TestConvertOpenAIResponsesRequestToClaude_PreservesContentPartCacheControl(t *testing.T) {
	inputJSON := `{
		"model": "gpt-4.1",
		"input": [
			{
				"type": "message",
				"role": "user",
				"content": [
					{"type": "input_text", "text": "cached prefix", "cache_control": {"type": "ephemeral"}},
					{"type": "input_text", "text": "fresh question"}
				]
			}
		]
	}`

	result := ConvertOpenAIResponsesRequestToClaude("claude-sonnet-4-5", []byte(inputJSON), false)
	resultJSON := gjson.ParseBytes(result)

	content := resultJSON.Get("messages.0.content")
	if !content.IsArray() {
		t.Fatalf("expected content array when cache_control is present, got %s", result)
	}
	if got := content.Get("0.cache_control.type").String(); got != "ephemeral" {
		t.Fatalf("content.0.cache_control.type = %q, want ephemeral. Output: %s", got, result)
	}
	if content.Get("1.cache_control").Exists() {
		t.Fatalf("content.1 should not have cache_control. Output: %s", result)
	}
}

func TestConvertOpenAIResponsesRequestToClaude_ReplaysReasoningBeforeFunctionCall(t *testing.T) {
	inputJSON := []byte(`{
		"model":"deepseek-v4-flash",
		"input":[
			{
				"type":"reasoning",
				"id":"rs_prev",
				"summary":[{"type":"summary_text","text":"previous thinking block"}]
			},
			{
				"type":"function_call",
				"call_id":"call_123",
				"name":"read_file",
				"arguments":"{\"path\":\"README.md\"}"
			},
			{
				"type":"function_call_output",
				"call_id":"call_123",
				"output":"file content"
			}
		]
	}`)

	result := ConvertOpenAIResponsesRequestToClaude("deepseek-v4-flash", inputJSON, true)

	messages := gjson.GetBytes(result, "messages")
	if !messages.Exists() || !messages.IsArray() {
		t.Fatalf("messages missing: %s", string(result))
	}
	if got := len(messages.Array()); got != 2 {
		t.Fatalf("messages len = %d, want 2: %s", got, string(result))
	}

	first := messages.Array()[0]
	if got := first.Get("role").String(); got != "assistant" {
		t.Fatalf("first role = %q, want assistant", got)
	}
	if got := first.Get("content.0.type").String(); got != "thinking" {
		t.Fatalf("first content[0].type = %q, want thinking", got)
	}
	if got := first.Get("content.0.thinking").String(); got != "previous thinking block" {
		t.Fatalf("first thinking = %q", got)
	}
	if got := first.Get("content.1.type").String(); got != "tool_use" {
		t.Fatalf("first content[1].type = %q, want tool_use", got)
	}
	if got := first.Get("content.1.id").String(); got != "call_123" {
		t.Fatalf("tool_use id = %q, want call_123", got)
	}

	second := messages.Array()[1]
	if got := second.Get("role").String(); got != "user" {
		t.Fatalf("second role = %q, want user", got)
	}
	if got := second.Get("content.0.type").String(); got != "tool_result" {
		t.Fatalf("second content[0].type = %q, want tool_result", got)
	}
	if got := second.Get("content.0.tool_use_id").String(); got != "call_123" {
		t.Fatalf("tool_result tool_use_id = %q, want call_123", got)
	}
}

func TestConvertOpenAIResponsesRequestToClaude_ReplaysReasoningAsAssistantMessage(t *testing.T) {
	inputJSON := []byte(`{
		"model":"deepseek-v4-flash",
		"input":[
			{
				"type":"reasoning",
				"id":"rs_prev",
				"summary":[{"type":"summary_text","text":"carry this thinking"}]
			},
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"continuing answer"}]
			}
		]
	}`)

	result := ConvertOpenAIResponsesRequestToClaude("deepseek-v4-flash", inputJSON, false)
	messages := gjson.GetBytes(result, "messages")
	if got := len(messages.Array()); got != 1 {
		t.Fatalf("messages len = %d, want 1: %s", got, string(result))
	}

	msg := messages.Array()[0]
	if got := msg.Get("role").String(); got != "assistant" {
		t.Fatalf("role = %q, want assistant", got)
	}
	if got := msg.Get("content.0.type").String(); got != "thinking" {
		t.Fatalf("content[0].type = %q, want thinking", got)
	}
	if got := msg.Get("content.0.thinking").String(); got != "carry this thinking" {
		t.Fatalf("thinking = %q", got)
	}
	if got := msg.Get("content.1.type").String(); got != "text" {
		t.Fatalf("content[1].type = %q, want text", got)
	}
	if got := msg.Get("content.1.text").String(); got != "continuing answer" {
		t.Fatalf("text = %q", got)
	}
}

func TestConvertOpenAIResponsesRequestToClaude_MergesAssistantTextAndToolUseIntoSameMessage(t *testing.T) {
	inputJSON := []byte(`{
		"model":"deepseek-v4-flash",
		"input":[
			{
				"type":"reasoning",
				"id":"rs_prev",
				"summary":[{"type":"summary_text","text":"carry this thinking"}]
			},
			{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"continuing answer"}]
			},
			{
				"type":"function_call",
				"call_id":"call_456",
				"name":"read_file",
				"arguments":"{\"path\":\"README.md\"}"
			}
		]
	}`)

	result := ConvertOpenAIResponsesRequestToClaude("deepseek-v4-flash", inputJSON, false)
	messages := gjson.GetBytes(result, "messages")
	if got := len(messages.Array()); got != 1 {
		t.Fatalf("messages len = %d, want 1: %s", got, string(result))
	}

	msg := messages.Array()[0]
	if got := msg.Get("role").String(); got != "assistant" {
		t.Fatalf("role = %q, want assistant", got)
	}
	if got := msg.Get("content.0.type").String(); got != "thinking" {
		t.Fatalf("content[0].type = %q, want thinking", got)
	}
	if got := msg.Get("content.1.type").String(); got != "text" {
		t.Fatalf("content[1].type = %q, want text", got)
	}
	if got := msg.Get("content.1.text").String(); got != "continuing answer" {
		t.Fatalf("content[1].text = %q, want continuing answer", got)
	}
	if got := msg.Get("content.2.type").String(); got != "tool_use" {
		t.Fatalf("content[2].type = %q, want tool_use", got)
	}
	if got := msg.Get("content.2.id").String(); got != "call_456" {
		t.Fatalf("content[2].id = %q, want call_456", got)
	}
}

func TestConvertOpenAIResponsesRequestToClaude_GroupsConsecutiveToolResultsIntoSameUserMessage(t *testing.T) {
	inputJSON := []byte(`{
		"model":"deepseek-v4-pro",
		"input":[
			{
				"type":"function_call",
				"call_id":"call_a",
				"name":"tool_a",
				"arguments":"{}"
			},
			{
				"type":"function_call",
				"call_id":"call_b",
				"name":"tool_b",
				"arguments":"{}"
			},
			{
				"type":"function_call_output",
				"call_id":"call_a",
				"output":"result a"
			},
			{
				"type":"function_call_output",
				"call_id":"call_b",
				"output":"result b"
			}
		]
	}`)

	result := ConvertOpenAIResponsesRequestToClaude("deepseek-v4-pro", inputJSON, false)
	messages := gjson.GetBytes(result, "messages")
	if got := len(messages.Array()); got != 2 {
		t.Fatalf("messages len = %d, want 2: %s", got, string(result))
	}

	first := messages.Array()[0]
	if got := first.Get("role").String(); got != "assistant" {
		t.Fatalf("first role = %q, want assistant", got)
	}
	if got := first.Get("content.0.type").String(); got != "tool_use" {
		t.Fatalf("first content[0].type = %q, want tool_use", got)
	}
	if got := first.Get("content.1.type").String(); got != "tool_use" {
		t.Fatalf("first content[1].type = %q, want tool_use", got)
	}

	second := messages.Array()[1]
	if got := second.Get("role").String(); got != "user" {
		t.Fatalf("second role = %q, want user", got)
	}
	if got := second.Get("content.0.type").String(); got != "tool_result" {
		t.Fatalf("second content[0].type = %q, want tool_result", got)
	}
	if got := second.Get("content.0.tool_use_id").String(); got != "call_a" {
		t.Fatalf("second content[0].tool_use_id = %q, want call_a", got)
	}
	if got := second.Get("content.1.type").String(); got != "tool_result" {
		t.Fatalf("second content[1].type = %q, want tool_result", got)
	}
	if got := second.Get("content.1.tool_use_id").String(); got != "call_b" {
		t.Fatalf("second content[1].tool_use_id = %q, want call_b", got)
	}
}
