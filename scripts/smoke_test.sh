#!/bin/bash
# =============================================================================
# Smoke Test: End-to-End Pipeline Validation (2 participants/group)
# =============================================================================
# Runs the full pipeline locally with a minimal participant count to verify
# that all scripts chain end-to-end. Uses 2 participants per group instead
# of 30 to keep total runtime under ~15 minutes.
#
# With 2 participants/group: 2 groups x 2 participants x 3 sessions = 12
# participant-sessions. Each MCMC fit takes ~10-30s, so fitting takes ~4-12min.
#
# Expected behaviour with 2 participants:
#   - 03_simulate: produces 12 participant-sessions (OK)
#   - 04_fit: fits all 12 sessions per model (OK)
#   - 05_validate: recovery skipped (needs >=30 participants), prints warning (OK)
#   - 06_group_analysis: group models skipped (needs >=6 participants),
#     but estimates_wide.csv and effect_sizes.csv are still produced (OK)
#
# Prerequisites:
#   - ds_env conda environment active (or prl_gpu on cluster)
#   - prl_hgf package installed (pip install -e .)
#
# Usage:
#   bash scripts/smoke_test.sh           # Run and clean up outputs
#   bash scripts/smoke_test.sh --keep    # Run and keep outputs
# =============================================================================

set -euo pipefail

# =============================================================================
# Parse Arguments
# =============================================================================
KEEP_OUTPUTS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep)
            KEEP_OUTPUTS=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash scripts/smoke_test.sh [--keep]"
            echo ""
            echo "Options:"
            echo "  --keep    Keep output files after smoke test"
            echo "  --help    Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================
# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

CONFIG="$PROJECT_ROOT/configs/prl_analysis.yaml"
CONFIG_BACKUP="$PROJECT_ROOT/configs/prl_analysis.yaml.bak"
TOTAL_START=$SECONDS

echo "============================================================"
echo "PRL HGF Smoke Test"
echo "============================================================"
echo "Project: $PROJECT_ROOT"
echo "Config: $CONFIG"
echo "Keep outputs: $KEEP_OUTPUTS"
echo "Start: $(date)"
echo "============================================================"

# =============================================================================
# Config Patch: 30 -> 2 participants/group (with safe restore on exit)
# =============================================================================
echo ""
echo "Patching config: n_participants_per_group 30 -> 2"
cp "$CONFIG" "$CONFIG_BACKUP"

# Trap to restore config on ANY exit (success, error, or signal)
restore_config() {
    if [[ -f "$CONFIG_BACKUP" ]]; then
        cp "$CONFIG_BACKUP" "$CONFIG"
        rm -f "$CONFIG_BACKUP"
        echo ""
        echo "Config restored to original (n_participants_per_group: 30)"
    fi
}
trap restore_config EXIT

# Patch the config
sed -i 's/n_participants_per_group: 30/n_participants_per_group: 2/' "$CONFIG"

# Verify patch applied
PATCHED_VALUE=$(grep "n_participants_per_group" "$CONFIG" | head -1 | awk '{print $2}')
if [[ "$PATCHED_VALUE" != "2" ]]; then
    echo "ERROR: Config patch failed. Expected 2, got: $PATCHED_VALUE"
    exit 1
fi
echo "Config patched: n_participants_per_group = $PATCHED_VALUE"

# =============================================================================
# Results tracking
# =============================================================================
declare -a STEP_NAMES=()
declare -a STEP_DURATIONS=()
declare -a STEP_STATUSES=()
OVERALL_STATUS="PASS"

run_step() {
    local step_name="$1"
    local step_cmd="$2"
    local expected_output="$3"

    echo ""
    echo "============================================================"
    echo "Step: $step_name"
    echo "============================================================"
    echo "Command: $step_cmd"
    echo "Started: $(date)"

    local step_start=$SECONDS

    # Run the command, capture exit code
    set +e
    eval "$step_cmd"
    local exit_code=$?
    set -e

    local step_duration=$(( SECONDS - step_start ))

    STEP_NAMES+=("$step_name")
    STEP_DURATIONS+=("${step_duration}s")

    if [[ $exit_code -ne 0 ]]; then
        echo ""
        echo "*** FAIL: $step_name exited with code $exit_code ***"
        STEP_STATUSES+=("FAIL (exit $exit_code)")
        OVERALL_STATUS="FAIL"
        return $exit_code
    fi

    # Check expected output file
    if [[ -n "$expected_output" && ! -e "$expected_output" ]]; then
        echo ""
        echo "*** FAIL: Expected output not found: $expected_output ***"
        STEP_STATUSES+=("FAIL (no output)")
        OVERALL_STATUS="FAIL"
        return 1
    fi

    if [[ -n "$expected_output" ]]; then
        local line_count
        line_count=$(wc -l < "$expected_output" 2>/dev/null || echo "0")
        echo "Output: $expected_output ($line_count lines)"
    fi

    STEP_STATUSES+=("PASS")
    echo "Duration: ${step_duration}s"
    return 0
}

# =============================================================================
# Pipeline Steps
# =============================================================================

# Step 1: Simulate
run_step \
    "03_simulate_participants" \
    "python scripts/03_simulate_participants.py" \
    "data/simulated/simulated_participants.csv"

# Step 2: Fit hgf_2level
run_step \
    "04_fit (hgf_2level)" \
    "python scripts/04_fit_participants.py --model hgf_2level" \
    "data/fitted/hgf_2level_results.csv"

# Step 3: Fit hgf_3level
run_step \
    "04_fit (hgf_3level)" \
    "python scripts/04_fit_participants.py --model hgf_3level" \
    "data/fitted/hgf_3level_results.csv"

# Step 4: Validation (recovery will be skipped due to <30 participants, that's OK)
run_step \
    "05_validation (--skip-waic)" \
    "python scripts/05_run_validation.py --skip-waic" \
    "results/validation"

# Step 5: Group analysis (bambi models will be skipped due to <6 participants, that's OK)
run_step \
    "06_group_analysis (2level)" \
    "python scripts/06_group_analysis.py --model 2level --skip-plots" \
    "results/group_analysis"

# =============================================================================
# Summary
# =============================================================================
TOTAL_DURATION=$(( SECONDS - TOTAL_START ))

echo ""
echo ""
echo "============================================================"
echo "SMOKE TEST SUMMARY"
echo "============================================================"
echo ""
printf "%-30s %-10s %-20s\n" "Step" "Duration" "Status"
printf "%-30s %-10s %-20s\n" "------------------------------" "----------" "--------------------"
for i in "${!STEP_NAMES[@]}"; do
    printf "%-30s %-10s %-20s\n" "${STEP_NAMES[$i]}" "${STEP_DURATIONS[$i]}" "${STEP_STATUSES[$i]}"
done
echo ""
echo "Total duration: ${TOTAL_DURATION}s ($(( TOTAL_DURATION / 60 ))m $(( TOTAL_DURATION % 60 ))s)"
echo "Overall status: $OVERALL_STATUS"
echo ""

# =============================================================================
# Cleanup (unless --keep)
# =============================================================================
if [[ "$KEEP_OUTPUTS" == "false" ]]; then
    echo "Cleaning up smoke test outputs..."
    rm -rf data/simulated/simulated_participants.csv
    rm -rf data/fitted/hgf_2level_results.csv
    rm -rf data/fitted/hgf_3level_results.csv
    rm -rf results/validation
    rm -rf results/group_analysis
    echo "Outputs cleaned."
else
    echo "Outputs kept (--keep flag)."
    echo "  data/simulated/simulated_participants.csv"
    echo "  data/fitted/hgf_2level_results.csv"
    echo "  data/fitted/hgf_3level_results.csv"
    echo "  results/validation/"
    echo "  results/group_analysis/"
fi

echo ""
echo "============================================================"
if [[ "$OVERALL_STATUS" == "PASS" ]]; then
    echo "SMOKE TEST PASSED"
else
    echo "SMOKE TEST FAILED"
fi
echo "============================================================"

# Exit with appropriate code
if [[ "$OVERALL_STATUS" != "PASS" ]]; then
    exit 1
fi
