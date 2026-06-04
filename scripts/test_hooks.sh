#!/usr/bin/env bash
# Smoke-test the v3 manuscript hooks with synthetic Claude Code payloads.
#
# Run from inside a scaffolded manuscript project (.claude/hooks/ must exist).
# Exits non-zero if any expected block/allow is wrong.

set -euo pipefail

HOOK_DIR=".claude/hooks"
if [[ ! -d "$HOOK_DIR" ]]; then
    echo "ERROR: $HOOK_DIR not found. Run this from a scaffolded manuscript project." >&2
    exit 2
fi

PROJ_ROOT="$(pwd)"
FAIL=0

run_case() {
    local name="$1"
    local hook="$2"
    local expect="$3"   # "block" or "allow"
    local payload="$4"
    actual_rc=0
    echo "$payload" | python3 "$HOOK_DIR/$hook" >/dev/null 2>&1 || actual_rc=$?
    if [[ "$expect" == "block" && "$actual_rc" -eq 2 ]] || \
       [[ "$expect" == "allow" && "$actual_rc" -eq 0 ]]; then
        echo "  ✓ $name"
    else
        echo "  ✗ $name (expected $expect, got rc=$actual_rc)"
        FAIL=$((FAIL + 1))
    fi
}

echo "▶ Testing pre_write_forbidden_tokens.py"
run_case "clean prose allowed" pre_write_forbidden_tokens.py allow \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$PROJ_ROOT/manuscript/intro.md\",\"content\":\"The cohort had 312 patients and a median follow-up of 5.2 years.\"}}"

run_case "AI-tell vocab blocked" pre_write_forbidden_tokens.py block \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$PROJ_ROOT/manuscript/discussion.md\",\"content\":\"This study delves into the intricate landscape of treatment.\"}}"

run_case "robust standard errors NOT blocked" pre_write_forbidden_tokens.py allow \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$PROJ_ROOT/manuscript/methods.md\",\"content\":\"We used robust standard errors and comprehensive genomic profiling on cases of essential thrombocythemia.\"}}"

run_case "null hypothesis NOT blocked" pre_write_forbidden_tokens.py allow \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$PROJ_ROOT/manuscript/methods.md\",\"content\":\"The null hypothesis assumed no association between exposure and outcome.\"}}"

run_case "null finding blocked" pre_write_forbidden_tokens.py block \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$PROJ_ROOT/manuscript/results.md\",\"content\":\"The result was null and no effect was seen.\"}}"

run_case "em-dash clause blocked" pre_write_forbidden_tokens.py block \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$PROJ_ROOT/manuscript/discussion.md\",\"content\":\"The cohort was large — including 312 patients — and well characterised.\"}}"

run_case "archive path exempt" pre_write_forbidden_tokens.py allow \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$PROJ_ROOT/archive/old.md\",\"content\":\"furthermore the multifaceted tapestry\"}}"

echo
echo "▶ Testing mandatory_delegation.py"
run_case "orchestrator reading figure → BLOCK" mandatory_delegation.py block \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Read\",\"tool_input\":{\"file_path\":\"$PROJ_ROOT/figures/fig1.png\"}}"

run_case "figure-visual-assessor reading figure → ALLOW" mandatory_delegation.py allow \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Read\",\"agent_type\":\"figure-visual-assessor\",\"tool_input\":{\"file_path\":\"$PROJ_ROOT/figures/fig1.png\"}}"

run_case "orchestrator writing manuscript → BLOCK" mandatory_delegation.py block \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$PROJ_ROOT/manuscript/intro.md\",\"content\":\"x\"}}"

run_case "manuscript-prose-writer writing manuscript → ALLOW" mandatory_delegation.py allow \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Write\",\"agent_type\":\"manuscript-prose-writer\",\"tool_input\":{\"file_path\":\"$PROJ_ROOT/manuscript/intro.md\",\"content\":\"x\"}}"

run_case "orchestrator bash redirect into manuscript → BLOCK" mandatory_delegation.py block \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"echo 'x' > manuscript/intro.md\"}}"

run_case "prose-writer bash redirect into manuscript → ALLOW" mandatory_delegation.py allow \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Bash\",\"agent_type\":\"manuscript-prose-writer\",\"tool_input\":{\"command\":\"echo 'x' > manuscript/intro.md\"}}"

run_case "bash pandoc to figures (data figure) → ALLOW" mandatory_delegation.py allow \
    "{\"cwd\":\"$PROJ_ROOT\",\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"Rscript -e 'ggsave' --out figures/fig1.png\"}}"

echo
echo "▶ Testing stop_status_check.py"

# Fresh log + matching marker
mkdir -p session_logs
echo "=== STATUS: Step 1 ===" > "session_logs/$(date +%Y-%m-%d_%H%M)_step1.md"

run_case "status block + matching fresh log → ALLOW" stop_status_check.py allow \
    "{\"cwd\":\"$PROJ_ROOT\",\"stop_hook_active\":false,\"last_assistant_message\":\"working...\\n=== STATUS: Step 1 ===\\nStep goal: test\\n======================\"}"

run_case "no status block → BLOCK" stop_status_check.py block \
    "{\"cwd\":\"$PROJ_ROOT\",\"stop_hook_active\":false,\"last_assistant_message\":\"done.\"}"

run_case "SKIP STATUS escape → ALLOW" stop_status_check.py allow \
    "{\"cwd\":\"$PROJ_ROOT\",\"stop_hook_active\":false,\"last_assistant_message\":\"quick question - [SKIP STATUS]\"}"

run_case "stop_hook_active=true → ALLOW (no loop)" stop_status_check.py allow \
    "{\"cwd\":\"$PROJ_ROOT\",\"stop_hook_active\":true}"

# Clean up
rm -f "session_logs/$(date +%Y-%m-%d_%H%M)_step1.md"

echo
if [[ "$FAIL" -gt 0 ]]; then
    echo "✗ $FAIL test(s) failed" >&2
    exit 1
fi
echo "✔ All hook tests passed."
