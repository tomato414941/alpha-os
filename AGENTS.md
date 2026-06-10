# AGENTS.md

Do not modify this file unless the user explicitly asks to change `AGENTS.md`.

Alpha work must prioritize finding many real candidates over building tracking
or research infrastructure.

## Alpha Work Rules

- The previous `strategies/` exploration workspace was removed intentionally
  (2026-06-10, commit d1bff82). Do not rebuild it or restore its pipelines
  from git history without an explicit user instruction.
- Do not add new methods, lanes, or probes while any candidate awaits a
  verdict. New probes are allowed only after completing a promote/kill
  decision on an existing candidate.
- Every new probe must first answer: who is on the other side of this trade,
  and why do we beat them given our constraints (capital, fees, latency,
  data access)?
- Method sophistication is not a substitute for statistical power or cost
  edge.
- Every candidate must log decisions and outcomes in one shared record schema
  (timestamp, symbol, direction, size, horizon, cost assumptions, net result)
  so evidence pools across candidates instead of living in bespoke files.
- No verdict may be claimed without a pre-registered rejection rule and
  enough observations to distinguish the result from noise.

## Repository Safety

- Get explicit approval before keeping branches, aliases, or deprecated
  interfaces for backward compatibility.
- Do not commit local runtime data, credentials, logs, or machine-specific
  deployment notes.

## Project Commands

- Use `uv` for project commands.
