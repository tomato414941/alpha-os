#!/bin/bash
# Check for deprecated patterns that should be removed.
# Run periodically to track migration progress.

echo "=== Deprecated Patterns ==="
echo

echo "--- State machine (remove after full stake migration) ---"
grep -rn "AlphaState\.\(ACTIVE\|DORMANT\|REJECTED\|CANDIDATE\)" src/alpha_os/ --include="*.py" | grep -v "test_" | wc -l
echo "  AlphaState references"

grep -rn "list_by_state\|update_state\|bulk_update_states" src/alpha_os/ --include="*.py" | wc -l
echo "  state query/update calls"

grep -rn "_prune_active_overflow" src/alpha_os/ --include="*.py" | wc -l
echo "  prune overflow references"

echo
echo "--- Naming debt (remove with state machine) ---"
grep -rn "managed_active\|registry_active\|n_registry_active" src/alpha_os/ --include="*.py" | wc -l
echo "  managed/registry active references"

grep -rn "deployed_alphas\|list_deployed" src/alpha_os/ --include="*.py" | wc -l
echo "  deployed_alphas references"

echo
echo "--- fitness_metric config (remove after full IC migration) ---"
grep -rn "fitness_metric\|config\.fitness_metric" src/alpha_os/ --include="*.py" | wc -l
echo "  fitness_metric references"

echo
echo "--- IC/RIC fallback shims ---"
grep -rn '"ic".*oos_sharpe\|"ric".*oos_sharpe' src/alpha_os/ --include="*.py" | wc -l
echo "  IC→Sharpe fallback entries"

echo
echo "--- BacktestEngine in admission ---"
grep -rn "BacktestEngine" src/alpha_os/daemon/admission.py | wc -l
echo "  BacktestEngine in admission"

echo
echo "--- Direct parse(expr).evaluate in pipeline consumers ---"
grep -rn "parse(.*\.evaluate(data)" src/alpha_os/daemon/admission.py src/alpha_os/paper/trader.py 2>/dev/null | wc -l
echo "  direct eval in admission/trader"

echo
echo "Target: all counts → 0"
