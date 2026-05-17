# Documentation Policy

## File Roles

- `AGENTS.md` - compact canonical agent bootstrap and reference index.
- `CLAUDE.md` - compact Claude bootstrap and reference index.
- `docs/agent/*` - progressive-disclosure operational knowledge.
- `tasks/lessons.md` - generalized anti-patterns and lessons.
- `tasks/todo.md` - task state.
- `README.md` - user-facing behavior and project overview.

## When Documenting a Lesson, Decision, or Fix

Check all of these in the same change:

- `AGENTS.md`, if it changes how agents should work.
- `CLAUDE.md`, if it changes Claude bootstrap behavior.
- `docs/agent/*`, if it is detailed operational context.
- `tasks/lessons.md`, if it is a reusable anti-pattern or lesson.
- `tasks/todo.md`, if task state changed.
- `README.md`, if user-facing behavior changed.

Do not update only one tracking file without checking the others.

## Context Budget Rule

Root files should contain routing and invariants, not the full runbook. Put large
domain sections in focused files under `docs/agent/` and reference them from the
root index.

## Backup Policy

For major root-context rewrites, back up `AGENTS.md` and `CLAUDE.md` under
`backup/` first. `backup/` is local-only and must not be pushed to GitHub.

