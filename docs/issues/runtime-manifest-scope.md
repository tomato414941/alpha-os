# Runtime Manifest Scope

## Problem

Runtime manifests are executable configuration bundles, but they can look like
the full research hypothesis.

They currently collect several run-time concerns in one file:

- subject sets
- observation requirements
- signal discovery configuration
- evaluation specs
- strategy specs
- evaluation tasks
- portfolio and cost assumptions

This is useful for reproducible runs, but it makes the manifest easy to treat as
the research source of truth.

## Current Decision

Keep the current manifest structure for now.

It is already connected to loaders, CLI commands, tests, README examples, and
golden paths. Splitting it now would create a broad infrastructure change before
a concrete evaluation requires it.

## Risk

New responsibilities may keep getting added to runtime manifests because they
are the easiest place to put experiment-related data.

That would make it harder to tell whether a field is needed to execute a run or
to explain the research idea behind the run.

## Guard

Do not add new top-level runtime manifest responsibilities unless they are
required to execute or reproduce a run.

The manifest may be the executable source of truth. It should not be treated as
the full research hypothesis.

## Later

Split runtime manifest responsibilities only when duplication or confusion
blocks a concrete evaluation.
