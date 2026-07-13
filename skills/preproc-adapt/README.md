# preproc-adapt skill

A Claude Code / agent skill for improving `deepcell-types` predictions on a
single FOV by adapting per-FOV image preprocessing — no retraining. It pairs
with the released model and its `predict(..., preprocess=...)` hook.

## Requirements

```bash
pip install "deepcell-types @ git+https://github.com/vanvalenlab/deepcell-types@master"
```

The skill uses the package's `predict`, `make_preprocessor`, and `DEFAULT_CONFIG`
APIs, so any install that includes them works.

## Install the skill

Copy (or symlink) this directory into your agent's skills directory:

- **Claude Code:** `~/.claude/skills/preproc-adapt/`
- **Codex:** `~/.agents/skills/preproc-adapt/`

```bash
# from a deepcell-types checkout
cp -r skills/preproc-adapt ~/.claude/skills/                       # Claude Code
# or symlink to track upstream updates:
ln -s "$PWD/skills/preproc-adapt" ~/.claude/skills/preproc-adapt
```

Restart the agent (or reload skills) so it discovers the skill. Then ask, e.g.,
*"my lymph-node FOV is called mostly Nerve — use preproc-adapt to fix it."*

## What it does

`SKILL.md` documents a closed loop: pre-register a panel-aware biological
expectation, run `predict`, roll predictions up to lineages, compare against the
frozen expectation, change one bounded preprocessing op via the `preprocess`
hook, and iterate — with guardrails that distinguish a fixable input artifact
from an unfixable panel/coverage gap.
