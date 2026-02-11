"""
Test each improvement branch in isolation.

For each feature/* branch:
1. Checkout the branch
2. Build with maturin develop --release
3. Run bench_engine.py benchmark
4. Run depth 6 vs 8 comparison (50 games)
5. Record results
6. Return to master

Usage:
    python scripts/test_branches.py              # Test all feature/* branches
    python scripts/test_branches.py --branch feature/lmr-table  # Test specific branch
    python scripts/test_branches.py --baseline   # Run baseline on master first
"""

import subprocess
import json
import os
import sys
import time
import argparse


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
RESULTS_FILE = os.path.join(SCRIPTS_DIR, 'improvement_results.json')


def run(cmd, cwd=PROJECT_DIR):
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True, timeout=600
    )
    return result.returncode, result.stdout, result.stderr


def get_feature_branches():
    """List all feature/* branches."""
    rc, out, _ = run('git branch --list "feature/*"')
    branches = [b.strip().lstrip('* ') for b in out.strip().split('\n') if b.strip()]
    return branches


def build_engine():
    """Build the Rust engine. Returns True on success."""
    print("  Building with maturin develop --release...")
    rc, out, err = run('maturin develop --release')
    if rc != 0:
        print(f"  BUILD FAILED: {err[:500]}")
        return False
    print("  Build successful.")
    return True


def run_bench(depth=8):
    """Run benchmark, return (total_nodes, total_time, nps)."""
    print(f"  Running benchmark @ depth {depth}...")
    python = os.path.join(PROJECT_DIR, '.venv', 'Scripts', 'python.exe')
    bench_script = os.path.join(SCRIPTS_DIR, 'bench_engine.py')
    rc, out, err = run(f'"{python}" "{bench_script}" --depth {depth}')
    if rc != 0:
        print(f"  BENCH FAILED: {err[:500]}")
        return None, None, None

    # Parse output for totals
    total_nodes = None
    total_time = None
    nps = None
    for line in out.split('\n'):
        if 'Total nodes:' in line:
            total_nodes = int(line.split(':')[1].strip().replace(',', ''))
        elif 'Total time:' in line:
            total_time = float(line.split(':')[1].strip().rstrip('s'))
        elif 'NPS:' in line:
            nps = int(line.split(':')[1].strip().replace(',', ''))

    return total_nodes, total_time, nps


def run_depth_match(games=50):
    """Run depth 6 vs 8 comparison. Returns (d6_wins, d8_wins, draws)."""
    print(f"  Running depth 6 vs 8 match ({games} games)...")
    python = os.path.join(PROJECT_DIR, '.venv', 'Scripts', 'python.exe')
    bench_script = os.path.join(SCRIPTS_DIR, 'bench_engine.py')
    rc, out, err = run(
        f'"{python}" "{bench_script}" --compare-depths --games {games}',
        cwd=PROJECT_DIR
    )
    if rc != 0:
        print(f"  MATCH FAILED: {err[:500]}")
        return None, None, None

    # Parse last line for result
    for line in reversed(out.split('\n')):
        if 'Result:' in line:
            # "Result: D6 X/50 (Y%) - D8 X/50 (Y%) - Draws Z"
            parts = line.split()
            d6_wins = int(parts[2].split('/')[0])
            d8_wins = int(parts[5].split('/')[0])
            draws = int(parts[-1])
            return d6_wins, d8_wins, draws

    return None, None, None


def test_branch(branch, baseline_nodes=None):
    """Test a single branch. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"Testing: {branch}")
    print(f"{'='*60}")

    # Checkout
    rc, _, err = run(f'git checkout {branch}')
    if rc != 0:
        print(f"  CHECKOUT FAILED: {err}")
        return {'branch': branch, 'status': 'checkout_failed'}

    # Build
    if not build_engine():
        run('git checkout master')
        return {'branch': branch, 'status': 'build_failed'}

    # Benchmark
    total_nodes, total_time, nps = run_bench(depth=8)

    # Depth match
    d6w, d8w, draws = run_depth_match(games=50)

    result = {
        'branch': branch,
        'status': 'ok',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'bench_depth': 8,
        'total_nodes': total_nodes,
        'total_time_s': total_time,
        'nps': nps,
        'depth_match': {
            'd6_wins': d6w,
            'd8_wins': d8w,
            'draws': draws,
        },
    }

    if baseline_nodes and total_nodes:
        diff = total_nodes - baseline_nodes
        pct = 100 * diff / baseline_nodes
        result['node_reduction_pct'] = round(pct, 1)
        print(f"\n  Node change vs baseline: {diff:+,} ({pct:+.1f}%)")

    if d6w is not None:
        print(f"  Depth match: D6 {d6w} - D8 {d8w} - Draws {draws}")

    # Return to master
    run('git checkout master')
    return result


def main():
    parser = argparse.ArgumentParser(description='Test improvement branches')
    parser.add_argument('--branch', type=str, help='Test specific branch')
    parser.add_argument('--baseline', action='store_true', help='Run baseline on master first')
    parser.add_argument('--skip-bench', action='store_true', help='Skip node benchmark')
    parser.add_argument('--games', type=int, default=50, help='Games for depth match')
    args = parser.parse_args()

    # Ensure we're on master to start
    run('git checkout master')

    # Load existing results
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    # Baseline
    baseline_nodes = results.get('master', {}).get('total_nodes')
    if args.baseline or not baseline_nodes:
        print("Running baseline on master...")
        if not build_engine():
            print("FATAL: Cannot build master!")
            sys.exit(1)
        total_nodes, total_time, nps = run_bench(depth=8)
        d6w, d8w, draws = run_depth_match(games=args.games)
        results['master'] = {
            'status': 'ok',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_nodes': total_nodes,
            'total_time_s': total_time,
            'nps': nps,
            'depth_match': {'d6_wins': d6w, 'd8_wins': d8w, 'draws': draws},
        }
        baseline_nodes = total_nodes
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline: {total_nodes:,} nodes, D6 {d6w}-{d8w}-{draws} D8")

    # Test branches
    if args.branch:
        branches = [args.branch]
    else:
        branches = get_feature_branches()

    if not branches:
        print("\nNo feature/* branches found. Create them first!")
        return

    print(f"\nBranches to test: {', '.join(branches)}")

    for branch in branches:
        result = test_branch(branch, baseline_nodes)
        results[branch] = result
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Branch':<30} {'Nodes':>12} {'Change':>8} {'D6':>3} {'D8':>3} {'Draw':>4}")
    print('-' * 60)

    if 'master' in results and results['master'].get('total_nodes'):
        r = results['master']
        dm = r.get('depth_match', {})
        print(f"{'master (baseline)':<30} {r['total_nodes']:>12,} {'---':>8} "
              f"{dm.get('d6_wins', '?'):>3} {dm.get('d8_wins', '?'):>3} {dm.get('draws', '?'):>4}")

    for branch in branches:
        r = results.get(branch, {})
        if r.get('status') != 'ok':
            print(f"{branch:<30} {'FAILED':>12}")
            continue
        dm = r.get('depth_match', {})
        node_change = f"{r.get('node_reduction_pct', 0):+.1f}%"
        print(f"{branch:<30} {r.get('total_nodes', 0):>12,} {node_change:>8} "
              f"{dm.get('d6_wins', '?'):>3} {dm.get('d8_wins', '?'):>3} {dm.get('draws', '?'):>4}")


if __name__ == '__main__':
    main()
