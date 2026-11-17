/**
 * Dictation research session extension (loaded only with `pi -e` by research.py).
 *
 * - `deliver_result`: the one way a result reaches the user's text field. It
 *   ends the session (terminate: true). The dictation service reads its
 *   arguments from the JSON event stream; free prose is never pasted.
 * - `tool_call` guard: policy.mjs decides; a refusal is a normal tool error the
 *   model sees, starting with BLOCK_PREFIX so the service can count it.
 *
 * - exit after shutdown: print mode ends by letting the event loop drain, so a
 *   handle another loaded extension keeps open would keep Pi running after
 *   the session ended. An unref'd timer exits the process a few seconds after
 *   `session_shutdown`; it never keeps Pi alive itself.
 *
 * Configuration comes from the environment the service sets
 * (DICTATION_RESEARCH_CWD / _NOTES / _PROPOSALS). Without it every write is
 * refused (fail closed).
 */
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { configFromEnv, decide, message } from "./policy.mjs";

const EXIT_AFTER_SHUTDOWN_MS = 3000;

export default function (pi: ExtensionAPI) {
	const cfg = configFromEnv(process.env);
	const state = { created: new Set<string>() };

	pi.registerTool({
		name: "deliver_result",
		label: "Deliver result",
		description:
			"Deliver the final result of this dictation research session. The text is pasted into the user's " +
			"text field (or put on the clipboard). Call it exactly once, as your last action.",
		promptSnippet: "Deliver the paste-ready result of the research session and end it",
		promptGuidelines: [
			"Always end a dictation research session with deliver_result, also when you found nothing (status no_result) or need the user to decide (status question).",
			"deliver_result.text is pasted as is: short, ready to use, in the language of the dictation, no greeting, no sign-off, no preamble such as 'Here is'.",
			"Call deliver_result on its own, never in the same turn as other tool calls: only then does it end the session at once.",
			"After calling deliver_result, do not emit another assistant response.",
		],
		parameters: Type.Object({
			text: Type.String({ description: "Paste-ready result, in the language of the dictation" }),
			summary: Type.String({ description: "One line: what was done and where the note is" }),
			status: StringEnum(["done", "no_result", "question"] as const, {
				description: "done: text answers the task; no_result: nothing useful found; question: the user must decide",
			}),
		}),
		async execute(_toolCallId, params) {
			return {
				content: [{ type: "text", text: "Result delivered; the session ends now." }],
				details: { status: params.status },
				terminate: true,
			};
		},
	});

	pi.on("session_shutdown", async (event) => {
		if (event.reason !== "quit") return;
		const timer = setTimeout(() => process.exit(process.exitCode ?? 0), EXIT_AFTER_SHUTDOWN_MS);
		timer.unref();
	});

	pi.on("tool_call", async (event) => {
		const decision = decide(event.toolName, event.input, cfg, state);
		if (decision.block) return { block: true, reason: message(decision) };
		return undefined;
	});
}
