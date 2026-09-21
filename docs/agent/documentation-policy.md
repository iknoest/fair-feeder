# Documentation policy

AGENTS.md is the shared project contract and task router. CLAUDE.md imports it;
Antigravity and Codex read it directly. Add tool-specific text only for an actual
capability or configuration difference, never a copied domain invariant.

Detailed procedures live in docs/agent/*. Task state belongs to tasks/todo.md;
lessons belong to tasks/lessons.md; user behavior belongs to README.md. Update a
file only when its role changed. Do not require parallel edits to all trackers.

Keep active instructions free of session history, tool-version snapshots and test
counts. Preserve useful domain gotchas in the relevant focused runbook. Major
bootstrap rewrites retain pre-change copies under local-only backup/; these are
historical and must not be pushed or read as current policy.
