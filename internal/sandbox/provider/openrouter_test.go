package provider

import (
	"strings"
	"testing"
)

func TestOpenRouterFactoryDefaults(t *testing.T) {
	pv, err := New(Config{Kind: KindOpenRouter})
	if err != nil {
		t.Fatalf("openrouter with no host/model: want defaults, got error: %v", err)
	}
	if pv == nil {
		t.Fatal("openrouter provider = nil, want non-nil")
	}
	op, ok := pv.(*OpenAIProvider)
	if !ok {
		t.Fatalf("openrouter kind = %T, want *OpenAIProvider", pv)
	}
	if op.cfg.UpstreamHost != openRouterUpstreamHost {
		t.Fatalf("openrouter default upstream host = %q, want %q", op.cfg.UpstreamHost, openRouterUpstreamHost)
	}
	if op.cfg.Model != defaultOpenRouterModel {
		t.Fatalf("openrouter default model = %q, want %q", op.cfg.Model, defaultOpenRouterModel)
	}
	if !strings.Contains(op.url, openRouterUpstreamHost+"/api/v1/chat/completions") {
		t.Fatalf("openrouter url = %q, want the %s /api/v1 path", op.url, openRouterUpstreamHost)
	}
}

func TestOpenRouterFactoryOverrides(t *testing.T) {
	const host = "gateway.example.test"
	const model = "meta-llama/llama-3.1-8b-instruct"

	pv, err := New(Config{Kind: "OpenRouter", UpstreamHost: host, Model: model})
	if err != nil {
		t.Fatalf("openrouter override: %v", err)
	}
	op, ok := pv.(*OpenAIProvider)
	if !ok {
		t.Fatalf("openrouter kind = %T, want *OpenAIProvider", pv)
	}
	if op.cfg.UpstreamHost != host {
		t.Fatalf("openrouter upstream host = %q, want overridden %q", op.cfg.UpstreamHost, host)
	}
	if op.cfg.Model != model {
		t.Fatalf("openrouter model = %q, want overridden %q", op.cfg.Model, model)
	}
	if !strings.Contains(op.url, host+"/v1/chat/completions") {
		t.Fatalf("openrouter override url = %q, want the overridden host standard /v1 path", op.url)
	}
}

package provider

import (
	"strings"
	"testing"
)

// TestOpenRouterPartialOverrideHostOnly verifies that providing only an
// UpstreamHost override while leaving Model empty still applies the default
// OpenRouter model.
func TestOpenRouterPartialOverrideHostOnly(t *testing.T) {
	const host = "custom-gateway.example.test"
	pv, err := New(Config{Kind: KindOpenRouter, UpstreamHost: host})
	if err != nil {
		t.Fatalf("openrouter partial override (host only): %v", err)
	}
	op, ok := pv.(*OpenAIProvider)
	if !ok {
		t.Fatalf("openrouter kind = %T, want *OpenAIProvider", pv)
	}
	// Host should be overridden.
	if op.cfg.UpstreamHost != host {
		t.Fatalf("upstream host = %q, want overridden %q", op.cfg.UpstreamHost, host)
	}
	// Model should fall back to the OpenRouter default.
	if op.cfg.Model != defaultOpenRouterModel {
		t.Fatalf("model = %q, want default %q", op.cfg.Model, defaultOpenRouterModel)
	}
	if !strings.Contains(op.url, host+"/api/v1/chat/completions") {
		t.Fatalf("url = %q, want overridden host with /api/v1 path", op.url)
	}
}

// TestOpenRouterPartialOverrideModelOnly verifies that providing only a
// Model override while leaving UpstreamHost empty still applies the default
// OpenRouter upstream host.
func TestOpenRouterPartialOverrideModelOnly(t *testing.T) {
	const model = "anthropic/claude-3.5-sonnet"
	pv, err := New(Config{Kind: KindOpenRouter, Model: model})
	if err != nil {
		t.Fatalf("openrouter partial override (model only): %v", err)
	}
	op, ok := pv.(*OpenAIProvider)
	if !ok {
		t.Fatalf("openrouter kind = %T, want *OpenAIProvider", pv)
	}
	// Host should fall back to the OpenRouter default.
	if op.cfg.UpstreamHost != openRouterUpstreamHost {
		t.Fatalf("upstream host = %q, want default %q", op.cfg.UpstreamHost, openRouterUpstreamHost)
	}
	// Model should be overridden.
	if op.cfg.Model != model {
		t.Fatalf("model = %q, want overridden %q", op.cfg.Model, model)
	}
	if !strings.Contains(op.url, openRouterUpstreamHost+"/api/v1/chat/completions") {
		t.Fatalf("url = %q, want default host with /api/v1 path", op.url)
	}
}

// TestOpenRouterRegistryDiscovery verifies that the OpenRouter kind is
// discoverable via the provider registry (i.e., a "openrouter" kind resolves
// to a factory that produces an *OpenAIProvider). It also checks that
// case-insensitive and whitespace-trimmed lookups work.
func TestOpenRouterRegistryDiscovery(t *testing.T) {
	kinds := []string{
		KindOpenRouter,
		"OpenRouter",
		" OPENROUTER ",
		"openRouter",
	}
	for _, kind := range kinds {
		t.Run(kind, func(t *testing.T) {
			pv, err := New(Config{Kind: kind})
			if err != nil {
				t.Fatalf("registry lookup for %q: %v", kind, err)
			}
			if pv == nil {
				t.Fatal("expected non-nil provider")
			}
			if _, ok := pv.(*OpenAIProvider); !ok {
				t.Fatalf("expected *OpenAIProvider, got %T", pv)
			}
		})
	}
}

// TestOpenRouterRegistryDiscovery alongside Anthropic verifies that
// registering OpenRouter does not displace the default Anthropic backend.
func TestOpenRouterCoexistsWithAnthropic(t *testing.T) {
	anthro, err := New(Config{Kind: KindAnthropic})
	if err != nil {
		t.Fatalf("anthropic lookup: %v", err)
	}
	if _, ok := anthro.(*AnthropicProvider); !ok {
		t.Fatalf("anthropic = %T, want *AnthropicProvider", anthro)
	}

	or, err := New(Config{Kind: KindOpenRouter})
	if err != nil {
		t.Fatalf("openrouter lookup: %v", err)
	}
	if _, ok := or.(*OpenAIProvider); !ok {
		t.Fatalf("openrouter = %T, want *OpenAIProvider", or)
	}
}