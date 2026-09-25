import assert from "node:assert/strict";
import { createAssistantMessageEventStream, InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { ModelRegistry, ModelRuntime } from "@earendil-works/pi-coding-agent";

import { createChildCompleter } from "../extensions/child-completion.ts";

function model(api, id, overrides = {}) {
	return {
		id,
		name: id,
		api,
		provider: "test-provider",
		baseUrl: "https://example.test",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 32_768,
		...overrides,
	};
}

const context = {
	systemPrompt: "child system prompt",
	messages: [
		{
			role: "user",
			content: [{ type: "text", text: "focused task" }],
			timestamp: Date.now(),
		},
	],
	tools: [],
};

const signal = new AbortController().signal;
// Exercise Pi's real dispatch, auth and context normalization. Only the remote
// provider is replaced; the adapter and model runtime both execute normally.
const runtime = await ModelRuntime.create({
	credentials: new InMemoryCredentialStore(),
	modelsPath: null,
	refreshOnCreate: false,
});
const registry = new ModelRegistry(runtime);
const cursorModel = model("cursor-native", "cursor-model", {
	provider: "test-cursor",
	thinkingLevelMap: { xhigh: "xhigh" },
});
const response = {
	role: "assistant",
	api: cursorModel.api,
	provider: cursorModel.provider,
	model: cursorModel.id,
	content: [{ type: "text", text: "CURSOR_OK" }],
	usage: {
		input: 11, output: 7, reasoning: 3, cacheRead: 0, cacheWrite: 0, totalTokens: 18,
		cost: { input: 0.11, output: 0.07, cacheRead: 0, cacheWrite: 0, total: 0.18 },
	},
	stopReason: "stop",
	timestamp: Date.now(),
};
const providerCalls = [];
let providerError;
registry.registerProvider(cursorModel.provider, {
	api: cursorModel.api,
	baseUrl: cursorModel.baseUrl,
	apiKey: "test-only-key",
	headers: { "X-Test-Provider": "registered" },
	models: [cursorModel],
	streamSimple(activeModel, activeContext, options) {
		providerCalls.push({ model: activeModel, context: activeContext, options });
		if (providerError) throw providerError;
		const stream = createAssistantMessageEventStream();
		stream.push({ type: "done", reason: "stop", message: response });
		stream.end();
		return stream;
	},
});
const registeredChild = createChildCompleter(registry);
for (const [activeModel, thinkingLevel, expectedReasoning] of [
	[cursorModel, "xhigh", "xhigh"],
	[cursorModel, "off", undefined],
	[{ ...cursorModel, reasoning: false }, "high", undefined],
]) {
	const result = await registeredChild({ model: activeModel, context, thinkingLevel, signal });
	assert.deepEqual(result, response);
	const call = providerCalls.at(-1);
	assert.equal(call.model, activeModel);
	// Pi 0.87 normalizes systemPrompt into a system message before dispatch.
	assert.deepEqual(call.context.messages, [
		{ role: "system", content: context.systemPrompt, timestamp: 0 },
		...context.messages,
	]);
	assert.equal(call.context.tools?.length ?? 0, 0);
	assert.equal(call.options.reasoning, expectedReasoning);
	assert.equal(call.options.signal, signal);
	assert.equal(call.options.apiKey, "test-only-key");
	assert.equal(call.options.headers["X-Test-Provider"], "registered");
	assert.equal(call.options.maxRetries, 0);
	assert.equal(call.options.cacheRetention, "none");
	assert.equal(call.options.sessionId, undefined);
}
assert.equal(providerCalls.length, 3);
providerError = new Error("custom provider failed");
const failed = await registeredChild({ model: cursorModel, context, thinkingLevel: "off", signal });
assert.equal(failed.stopReason, "error");
assert.match(failed.errorMessage, /custom provider failed/);
registry.unregisterProvider(cursorModel.provider);
const missing = await registeredChild({ model: cursorModel, context, thinkingLevel: "off", signal });
assert.equal(missing.stopReason, "error");
assert.match(missing.errorMessage, /Unknown provider: test-cursor/);
assert.equal(providerCalls.length, 4);
