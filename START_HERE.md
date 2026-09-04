# START HERE — how to drive this project with Claude Code

## 0. One-time setup (5 minutes)

```bash
# install Claude Code (needs Node 18+)
npm install -g @anthropic-ai/claude-code

cd "C:\Claude\AI Project"

# install the Claude Code config (slash commands + subagents + settings)
powershell -ExecutionPolicy Bypass -File claude-config\install.ps1

git init && git add -A && git commit -m "scaffold: research spec, roadmap, claude config"

claude          # first run authenticates
```

> The `claude-config\` folder is just a staging copy — `install.ps1` places its
> contents into `.claude\`, which is where Claude Code actually reads them.
> Re-run it any time you edit a command or subagent in `claude-config\`.

Everything Claude Code needs is already here:

```
CLAUDE.md              auto-loaded every session — the rules
docs/                  spec, architecture, roadmap, datasets, eval, ADRs
.claude/commands/      your slash commands: /phase /run-eval /ablate /checkpoint /paper-table
.claude/agents/        specialist subagents Claude delegates to
.claude/settings.json  permissions + auto-format hook
```

## 1. The loop you will actually run

```
/phase 0            → Claude plans Phase 0, you approve, it builds, exit test runs
/checkpoint         → updates roadmap status + ADRs, commits
                      (start a new session — keeps context clean)
/phase 1
/checkpoint
...
/run-eval medqa V5  → a real scored run into results/
/ablate             → the full grid from docs/02_ROADMAP.md Phase 7
/paper-table        → regenerates paper/tables from results/
```

**First message of the very first session** — paste this:

> Read CLAUDE.md and docs/00–05. Then run `/phase 0`. Before writing any code,
> give me the plan: file list, key interfaces (function signatures only), the
> exit test, and anything in the spec you think is wrong or under-specified.
> Do not write code until I approve the plan.

## 2. Habits that make the difference

- **Plan mode for anything non-trivial.** `Shift+Tab` twice, or use `/phase`,
  which forces it. Approve the plan, then let it run.
- **One phase per session.** `/clear` between phases; the roadmap + ADRs are the
  memory, not the transcript.
- **Let it delegate.** The subagents in `.claude/agents/` keep long file dumps
  (index builds, eval logs) out of your main context.
- **`#` to teach it.** Typing `# always use uv, never pip` writes the rule into
  CLAUDE.md permanently.
- **`Esc` early.** Interrupt the moment it heads the wrong way; cheaper than
  reviewing 400 wrong lines.
- **Long jobs → scripts.** Index builds and full evals run as `python -m scripts.*`
  with logs, so a dropped session never loses a run.
- **Give it eyes.** Screenshot a reliability diagram or a Neo4j subgraph and paste
  it in; it debugs plots and graph shapes far better with the image.

## 3. What runs where

| Work | Where |
|---|---|
| Writing code, refactors, tests, docs | Claude Code in this folder |
| Long GPU/index/eval runs | your machine or a GPU box, launched by Claude Code as scripts |
| Neo4j + PrimeKG | Docker, `make kg-up` |
| Reading papers, comparing to MEGA-RAG, drafting the write-up | this chat (Cowork) — it has web access and can write into this folder too |

Use this chat for the *research* thinking, use Claude Code for the *repository*.
Both read the same `docs/` — keep the docs the source of truth and neither one
drifts.

## 4. MCP servers worth adding later

```bash
claude mcp add neo4j -- npx -y @neo4j/mcp-server-cypher   # query the KG from chat
```
Add only when you actually need them; every server costs context.
