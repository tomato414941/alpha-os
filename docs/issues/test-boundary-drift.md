# Test Boundary Drift

## Problem

Some test files already cover several contracts instead of one clear boundary.

Examples include broad files that mix CLI behavior, manifest application,
runtime smoke tests, evaluation planning, and domain contracts.

## Current Decision

Do not reorganize the existing test tree yet.

The current suite passes, and splitting large files without a concrete blocking
change would create churn.

## Risk

New tests can keep being added to broad files simply because they already
exist.

Over time, tests stop acting like a map of system contracts and become a
warehouse of related scenarios.

## Guard

For new tests, prefer a small file named after the contract being protected.

Do not use an existing large test file as the default destination unless the
new test clearly belongs to the same contract.

## Next Decision

When adding a test that does not clearly belong to an existing focused file,
create a new small test file for that contract instead of extending a broad
test file by default.

## Close Condition

Close this when broad test files are no longer the default destination for new
contract tests, or when the remaining broad files have been split because a
concrete change required it.

## Later

Split existing large test files only when a concrete change is blocked or made
risky by their current shape.
