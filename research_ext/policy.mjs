// Tool guard for dictation research sessions (see docs/research.md).
//
// A denylist, deliberately: the session keeps reading, searching, the web and
// computing. Blocked: writes outside its own notes directory (plus, when
// allowed, a new file in an issues/ folder or an insert-only PLAN.md edit),
// deleting or overwriting, credential paths, external writes, service and
// process control, account and role changes. A denylist is never complete; the
// fixed session brief says the same in words.
//
// Pure functions, no Pi imports: tests run this file under plain node.
import fs from "node:fs";
import path from "node:path";

export const BLOCK_PREFIX = "Blocked by the dictation research guard";

// Tools that act outside the machine, change accounts/roles, drive the user's
// browser, wait for a human, or start other agents.
const DENY_TOOLS = new Set([
	"update_roles", "ask_user", "cockpit_claim", "canvas",
	"browser_connect", "browser_show_in_pane", "browser_await_human",
]);
const DENY_TOOL_PATTERNS = [
	/^mcp__/, /^browser_/, /^gmail_/, /^craft_/,
	/(^|_)(send|spawn|delete|remove|trash|push|publish|upload|post|reply|draft|dispatch|abort|deploy|modify|create|set|update|kill|restart)(_|$)/,
];

// Paths whose content is a credential or a key. Matched on the raw and the real path.
const SECRET_PATH = new RegExp(
	[
		"(^|/)(credentials|\\.ssh|\\.gnupg|\\.aws|\\.password-store|keyrings?|\\.docker)(/|$)",
		"(^|/)(auth|mcp-auth|plugin-credentials)\\.json$",
		"(^|/)\\.env(\\.[^/]*)?$",
		"\\.(pem|key|p12|pfx|kdbx|keystore)$",
		"(^|/)id_(rsa|ed25519|ecdsa|dsa)",
		"(^|/)\\.(netrc|git-credentials|pgpass)$",
		"(^|/)secrets?\\.(ya?ml|json|env|txt)$",
	].join("|"),
	"i",
);
// The same, inside a shell command (anywhere in the text).
const SECRET_IN_COMMAND = new RegExp(
	[
		"(^|[\\s/'\"=:])(credentials|\\.ssh|\\.gnupg|\\.aws|\\.password-store|keyrings?)(/|[\\s'\"]|$)",
		"(auth|mcp-auth|plugin-credentials)\\.json",
		"(^|[\\s/'\"])\\.env([\\s'\".]|$)",
		"\\.(pem|kdbx|p12|pfx)([\\s'\"]|$)",
		"id_(rsa|ed25519|ecdsa|dsa)",
		"\\.(netrc|git-credentials|pgpass)",
	].join("|"),
	"i",
);

// Command position: start of the command, after ; & | ( ` $( or a newline,
// or at the start of the code of `sh -c '...'`, optionally behind wrappers
// (env, VAR=value, xargs, nohup, time, nice, timeout N, command, exec).
const CMD = "(?:^|[;&|(`\\n]|\\$\\(|-c\\s+['\"])\\s*(?:(?:env|nohup|time|nice|ionice|stdbuf|xargs|command|builtin|exec|timeout\\s+\\S+|[A-Za-z_][A-Za-z0-9_]*=\\S*)\\s+)*(?:[\\w.~/-]*/)?";
const END = "(?=$|[\\s;&|)`'\"])";

// Programs never run from a research session: deleting, moving or copying
// over files, remote shells and copies, services, privileges, signals,
// permissions, clipboard and input injection, desktop settings, package
// managers, cloud and account CLIs, the credential vault, and pi itself
// (pi auth prints tokens).
const DENY_PROGRAMS = [
	["delete", "rm|rmdir|unlink|shred|truncate|wipe|srm"],
	["move", "mv|rename"],
	["copy", "cp|ln|install|rsync|tee|dd|touch"],
	["remote", "ssh|scp|sftp|ftp|telnet|nc|ncat|netcat|socat|mosh"],
	["service", "systemctl|service|loginctl|shutdown|reboot|poweroff|halt|crontab|launchctl"],
	["privilege", "sudo|su|doas|pkexec|runuser"],
	["signal", "kill|pkill|killall|xkill|skill"],
	["permission", "chmod|chown|chgrp|setfacl|chattr"],
	["desktop", "copyq|xdotool|ydotool|wtype|xclip|xsel|wl-copy|gdbus|dbus-send|busctl|gsettings|dconf|notify-send"],
	["package", "apt|apt-get|dpkg|snap|flatpak|pip|pip3|pipx|uv|npm|npx|pnpm|yarn|brew|gem|cargo"],
	["account", "pi|bw|secret-tool|gh|glab|gcloud|gsutil|aws|az|docker|podman|kubectl|terraform|pb|clasp|rclone|restic"],
	["disk", "mkfs|mkfs\\.[a-z0-9]+|fdisk|parted|mount|umount|losetup"],
];
const PROGRAM_RULES = DENY_PROGRAMS.map(([rule, names]) => [rule, new RegExp(`${CMD}(?:${names})${END}`)]);

// git: the first word after the global options is the subcommand. Reading
// (status, log, diff, show, grep, blame, ls-files) stays allowed.
const GIT = `${CMD}git(?:\\s+(?:-[cC]\\s+\\S+|--?[\\w-]+(?:=\\S+)?))*\\s+`;
const ARG_RULES = [
	["git_remote", new RegExp(`${GIT}(?:push|pull|fetch|clone|remote|submodule|send-email|request-pull|svn|daemon)\\b`)],
	["git_write", new RegExp(`${GIT}(?:add|commit|reset|clean|checkout|restore|stash|rebase|merge|rm|mv|revert|cherry-pick|am|apply|gc|prune|filter-branch|filter-repo|update-ref|update-index|switch|config|worktree|notes|replace|init|tag|branch\\s+-|reflog\\s+(?:expire|delete))\\b`)],
	["find_exec", /\bfind\b[^;&|\n]*\s-(delete|exec|execdir|ok|okdir|fprint|fprintf|fls)\b/],
	["in_place", /\b(sed|perl|ruby)\b[^;&|\n]*\s-[a-zA-Z]*i/],
	["http_write", /\bcurl\b[^;&|\n]*(\s-[a-zA-Z]*[XdFTo]\b|\s--(data[a-z-]*|form[a-z-]*|upload-file|request|json|post[a-z-]*|output)\b|\s-O\b)/],
	["http_write", /\bwget\b[^;&|\n]*(--(post|method|body|header)|\s-O\s|--output-document)/],
	["script_write", /\b(os\.(remove|unlink|rmdir|removedirs|rename|replace|kill|killpg|system|popen|exec\w*|spawn\w*|chmod|chown|truncate)|shutil\.(rmtree|move|copy\w*)|subprocess|requests\.(post|put|delete|patch)|httpx\.(post|put|delete|patch)|urlopen|urllib\.request|socket\.|smtplib|ftplib|paramiko|\.unlink\(|\.rmdir\(|\.rename\(|\.replace\(|\.write_text\(|\.write_bytes\(|\.touch\(|open\([^)]*['"](w|a|x|r\+)[bt+]*['"]|writeFile|appendFile|unlinkSync|rmSync|rmdirSync|renameSync|child_process|execSync|spawnSync|fetch\([^)]*method)/],
	["eval", new RegExp(`${CMD}(?:eval|source|\\.)\\s`)],
];

function isInside(child, parent) {
	const rel = path.relative(parent, child);
	return rel === "" || (!!rel && !rel.startsWith("..") && !path.isAbsolute(rel));
}

// Real path of a possibly non-existing target: real nearest ancestor + the rest.
export function realTarget(target) {
	let current = target;
	const rest = [];
	for (;;) {
		try {
			return path.join(fs.realpathSync(current), ...rest.reverse());
		} catch {
			const parent = path.dirname(current);
			if (parent === current) return target;
			rest.push(path.basename(current));
			current = parent;
		}
	}
}

export function configFromEnv(env) {
	const cwd = env.DICTATION_RESEARCH_CWD;
	const notes = env.DICTATION_RESEARCH_NOTES;
	if (!cwd || !notes || !path.isAbsolute(cwd) || !path.isAbsolute(notes)) return null;
	return {
		cwd: realTarget(cwd),
		notes: realTarget(notes),
		proposals: env.DICTATION_RESEARCH_PROPOSALS === "1",
	};
}

function block(rule, reason) {
	return { block: true, rule, reason };
}
const ALLOW = { block: false };

function resolveIn(cfg, p) {
	const base = cfg ? cfg.cwd : process.cwd();
	let value = String(p);
	if (value.startsWith("@")) value = value.slice(1);
	if (value === "~" || value.startsWith("~/")) value = path.join(process.env.HOME || "/", value.slice(1));
	return path.resolve(base, value);
}

function secretPath(cfg, p) {
	if (typeof p !== "string" || !p) return false;
	const abs = resolveIn(cfg, p);
	return SECRET_PATH.test(p) || SECRET_PATH.test(abs) || SECRET_PATH.test(realTarget(abs));
}

function pathArgs(input) {
	const out = [];
	for (const key of ["path", "file_path", "filePath", "file", "dir", "directory", "cwd", "glob", "pattern_path"]) {
		if (typeof input?.[key] === "string") out.push(input[key]);
	}
	for (const key of ["paths", "files"]) {
		if (Array.isArray(input?.[key])) for (const v of input[key]) if (typeof v === "string") out.push(v);
	}
	return out;
}

function editsOf(input) {
	if (Array.isArray(input?.edits)) return input.edits;
	if (typeof input?.oldText === "string" || typeof input?.old_string === "string") {
		return [{ oldText: input.oldText ?? input.old_string, newText: input.newText ?? input.new_string }];
	}
	return null;
}

function decideWrite(tool, input, cfg, state) {
	const raw = input?.path ?? input?.file_path;
	if (typeof raw !== "string" || !raw) return block("write_outside", "no path");
	if (!cfg) return block("write_outside", "guard not configured: no writes");
	const target = realTarget(resolveIn(cfg, raw));
	if (SECRET_PATH.test(target)) return block("secret_path", "credential path");
	const exists = fs.existsSync(target);
	const own = state.created.has(target);
	if (isInside(target, cfg.notes) && target !== cfg.notes) {
		if (tool === "write") {
			if (exists && !own) return block("overwrite", "existing file; write a new note instead");
			state.created.add(target);
			return ALLOW;
		}
		if (own || !exists) return ALLOW;
		return block("edit_outside", "only notes written in this session can be edited");
	}
	if (!isInside(target, cfg.cwd)) return block("write_outside", "outside the working directory");
	if (own) return ALLOW; // a proposal file written earlier in this session
	if (!cfg.proposals) return block("write_outside", "only the notes directory is writable");
	if (tool === "write") {
		if (path.basename(path.dirname(target)) === "issues" && target.endsWith(".md")) {
			if (exists) return block("overwrite", "existing issue file");
			state.created.add(target);
			return ALLOW;
		}
		return block("write_outside", "only the notes directory or a new file in an issues/ folder");
	}
	// edit: insert-only into a PLAN.md (a Beslissen line); nothing removed or changed.
	if (path.basename(target) === "PLAN.md" && exists) {
		const edits = editsOf(input);
		if (!edits || !edits.length) return block("edit_not_insert", "no edits");
		for (const e of edits) {
			if (typeof e?.oldText !== "string" || typeof e?.newText !== "string" || !e.oldText
				|| !e.newText.includes(e.oldText) || e.newText.length <= e.oldText.length) {
				return block("edit_not_insert", "PLAN.md edits may only insert text around the old text, never change it");
			}
		}
		return ALLOW;
	}
	return block("edit_outside", "only notes, new issue files and insert-only PLAN.md edits");
}

function redirectTargets(command) {
	const targets = [];
	const re = /(^|[^0-9&<>|])(?:[0-9]*|&)>>?(?!&)\s*("[^"]*"|'[^']*'|[^\s;&|()<>]+)/g;
	let m;
	while ((m = re.exec(command)) !== null) targets.push(m[2].replace(/^["']|["']$/g, ""));
	return targets;
}

function decideBash(input, cfg) {
	const command = typeof input?.command === "string" ? input.command : "";
	if (!command.trim()) return ALLOW;
	if (SECRET_IN_COMMAND.test(command)) return block("secret_path", "credential path in command");
	for (const [rule, re] of PROGRAM_RULES) if (re.test(command)) return block(rule, "program not allowed");
	for (const [rule, re] of ARG_RULES) if (re.test(command)) return block(rule, "command not allowed");
	for (const target of redirectTargets(command)) {
		if (/^\/dev\/(null|stdout|stderr)$/.test(target)) continue;
		const abs = realTarget(resolveIn(cfg, target));
		if (cfg && isInside(abs, cfg.notes) && abs !== cfg.notes && !fs.existsSync(abs)) continue;
		if (isInside(abs, "/tmp") && !fs.existsSync(abs)) continue;
		return block("redirect_outside", "shell writes only to a new file in the notes directory or /tmp");
	}
	return ALLOW;
}

// decide(toolName, input, cfg, state) -> {block:false} | {block:true, rule, reason}
// cfg: configFromEnv(...) or null (fail closed for writes); state: {created: Set}.
export function decide(toolName, input, cfg, state = { created: new Set() }) {
	const tool = String(toolName || "");
	if (DENY_TOOLS.has(tool) || DENY_TOOL_PATTERNS.some((re) => re.test(tool))) {
		if (tool !== "deliver_result") return block("tool_denied", "tool not allowed in a research session");
	}
	if (tool === "write" || tool === "edit") return decideWrite(tool, input, cfg, state);
	if (tool === "bash") return decideBash(input, cfg);
	for (const p of pathArgs(input)) if (secretPath(cfg, p)) return block("secret_path", "credential path");
	return ALLOW;
}

export function message(decision) {
	return `${BLOCK_PREFIX} [${decision.rule}]: ${decision.reason}. Do not try to work around it; mention what you needed in your result.`;
}
