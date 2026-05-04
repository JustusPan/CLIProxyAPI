package executor

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	"github.com/tidwall/gjson"
)

func TestApplyDeepSeekResponsesChatCompatibility_ReplaysReasoningBeforeToolCall(t *testing.T) {
	original := []byte(`{
		"model":"deepseek-v4-flash",
		"reasoning":{"effort":"medium"},
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
	translated := []byte(`{
		"model":"deepseek-v4-flash",
		"messages":[
			{
				"role":"assistant",
				"tool_calls":[{"id":"call_123","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"README.md\"}"}}]
			},
			{"role":"tool","tool_call_id":"call_123","content":"file content"}
		],
		"stream":true
	}`)

	out := applyDeepSeekResponsesChatCompatibility(translated, original)

	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != "previous thinking block" {
		t.Fatalf("messages.0.reasoning_content = %q, want previous thinking block\npayload: %s", got, string(out))
	}
	if got := gjson.GetBytes(out, "messages.0.content").String(); got != "" {
		t.Fatalf("messages.0.content = %q, want empty string", got)
	}
	if got := gjson.GetBytes(out, "messages.0.tool_calls.0.id").String(); got != "call_123" {
		t.Fatalf("messages.0.tool_calls.0.id = %q, want call_123", got)
	}
	if got := gjson.GetBytes(out, "thinking.type").String(); got != "enabled" {
		t.Fatalf("thinking.type = %q, want enabled", got)
	}
}

func TestOpenAICompatExecutorDeepSeekCompatibility_DefaultOff(t *testing.T) {
	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{}}
	translated := []byte(`{"messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]}]}`)
	original := []byte(`{"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]},{"type":"function_call","call_id":"call_1","name":"read","arguments":"{}"}]}`)

	out := executor.applyDeepSeekThinkingCompatibility(auth, translated, original, "openai-response", "gpt-5.2")

	if gjson.GetBytes(out, "messages.0.reasoning_content").Exists() {
		t.Fatalf("reasoning_content should not be injected when compatibility is disabled: %s", string(out))
	}
	if gjson.GetBytes(out, "thinking").Exists() {
		t.Fatalf("thinking should not be injected when compatibility is disabled: %s", string(out))
	}
}

func TestOpenAICompatExecutorDeepSeekCompatibility_EnabledByDeepSeekV4Model(t *testing.T) {
	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{}}
	translated := []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]}]}`)
	original := []byte(`{"reasoning":{"effort":"high"},"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]},{"type":"function_call","call_id":"call_1","name":"read","arguments":"{}"}]}`)

	out := executor.applyDeepSeekThinkingCompatibility(auth, translated, original, "openai-response")

	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != "thinking" {
		t.Fatalf("messages.0.reasoning_content = %q, want thinking", got)
	}
	if got := gjson.GetBytes(out, "thinking.type").String(); got != "enabled" {
		t.Fatalf("thinking.type = %q, want enabled", got)
	}
}

func TestOpenAICompatExecutorDeepSeekCompatibility_EnabledByRequestedDeepSeekV4ModelWithSuffix(t *testing.T) {
	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{}}
	translated := []byte(`{"model":"provider-alias","messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]}]}`)
	original := []byte(`{"reasoning":{"effort":"high"},"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]},{"type":"function_call","call_id":"call_1","name":"read","arguments":"{}"}]}`)

	out := executor.applyDeepSeekThinkingCompatibility(auth, translated, original, "openai-response", "deepseek-v4-pro(8192)")

	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != "thinking" {
		t.Fatalf("messages.0.reasoning_content = %q, want thinking", got)
	}
	if got := gjson.GetBytes(out, "thinking.type").String(); got != "enabled" {
		t.Fatalf("thinking.type = %q, want enabled", got)
	}
}

func TestOpenAICompatExecutorDeepSeekCompatibility_ReplaysReasoningForGroupedToolCalls(t *testing.T) {
	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{}}
	translated := []byte(`{
		"model":"deepseek-v4-pro",
		"messages":[
			{
				"role":"assistant",
				"tool_calls":[
					{"id":"call_a","type":"function","function":{"name":"tool_a","arguments":"{}"}},
					{"id":"call_b","type":"function","function":{"name":"tool_b","arguments":"{}"}}
				]
			},
			{"role":"tool","tool_call_id":"call_a","content":"result a"},
			{"role":"tool","tool_call_id":"call_b","content":"result b"}
		]
	}`)
	original := []byte(`{
		"reasoning":{"effort":"high"},
		"input":[
			{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking before tools"}]},
			{"type":"function_call","call_id":"call_a","name":"tool_a","arguments":"{}"},
			{"type":"function_call","call_id":"call_b","name":"tool_b","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_a","output":"result a"},
			{"type":"function_call_output","call_id":"call_b","output":"result b"}
		]
	}`)

	out := executor.applyDeepSeekThinkingCompatibility(auth, translated, original, "openai-response")

	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != "thinking before tools" {
		t.Fatalf("messages.0.reasoning_content = %q, want thinking before tools\npayload: %s", got, string(out))
	}
	if got := gjson.GetBytes(out, "messages.1.role").String(); got != "tool" {
		t.Fatalf("messages.1.role = %q, want tool", got)
	}
	if got := gjson.GetBytes(out, "messages.2.tool_call_id").String(); got != "call_b" {
		t.Fatalf("messages.2.tool_call_id = %q, want call_b", got)
	}
}

func TestOpenAICompatExecutorDeepSeekCompatibility_EnabledByConfig(t *testing.T) {
	executor := NewOpenAICompatExecutor("packyapi", &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{{
			Name:                          "PackyAPI",
			DeepSeekThinkingCompatibility: true,
		}},
	})
	auth := &cliproxyauth.Auth{
		Provider: "packyapi",
		Attributes: map[string]string{
			"compat_name": "PackyAPI",
		},
	}
	translated := []byte(`{"messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]}]}`)
	original := []byte(`{"reasoning":{"effort":"high"},"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]},{"type":"function_call","call_id":"call_1","name":"read","arguments":"{}"}]}`)

	out := executor.applyDeepSeekThinkingCompatibility(auth, translated, original, "openai-response", "gpt-5.2")

	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != "thinking" {
		t.Fatalf("messages.0.reasoning_content = %q, want thinking", got)
	}
	if got := gjson.GetBytes(out, "thinking.type").String(); got != "enabled" {
		t.Fatalf("thinking.type = %q, want enabled", got)
	}
}

func TestOpenAICompatExecutorDeepSeekCompatibility_NonResponsesSourceUntouched(t *testing.T) {
	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"deepseek_thinking_compatibility": "true"}}
	translated := []byte(`{"messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]}]}`)
	original := []byte(`{"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]},{"type":"function_call","call_id":"call_1","name":"read","arguments":"{}"}]}`)

	out := executor.applyDeepSeekThinkingCompatibility(auth, translated, original, "openai", "deepseek-v4-flash")

	if string(out) != string(translated) {
		t.Fatalf("non-responses source should be untouched, got: %s", string(out))
	}
}
