import { clampThinkingLevel } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { CompleteChild } from "./rlm-host.ts";

type ModelRegistry = ExtensionContext["modelRegistry"];

// Pi owns provider-specific reasoning, token budgets, credentials and routing.
// Use its public, provider-neutral seam rather than importing pi-ai internals
// from a second installation or reproducing every provider's option mapping.
export function createChildCompleter(modelRegistry: ModelRegistry): CompleteChild {
	return async ({ model, context, thinkingLevel, signal }) => {
		const level = model.reasoning ? clampThinkingLevel(model, thinkingLevel) : "off";
		return modelRegistry.streamSimple(model, context, {
			signal,
			maxRetries: 0,
			cacheRetention: "none",
			reasoning: level === "off" ? undefined : level,
		}).result();
	};
}
