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

## Later

Split existing large test files only when a concrete change is blocked or made
risky by their current shape.
