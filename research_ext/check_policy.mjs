// Test helper: one JSON case per stdin line -> one JSON decision per stdout line.
// Case: {"tool": "...", "input": {...}, "env": {...}}; cases share one state
// (files created by earlier writes), as in one session.
import { configFromEnv, decide } from "./policy.mjs";

const state = { created: new Set() };
let data = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { data += chunk; });
process.stdin.on("end", () => {
	for (const line of data.split("\n")) {
		if (!line.trim()) continue;
		const c = JSON.parse(line);
		const d = decide(c.tool, c.input, configFromEnv(c.env || {}), state);
		process.stdout.write(JSON.stringify(d) + "\n");
	}
});
