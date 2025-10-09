#!/usr/bin/env python3

"""
Run gradient descent experiments with automatic energy logging.

This script runs the gradient descent solvers with predefined tau cycles and 
automatically generates energy logs with meaningful names.
"""

import subprocess
import sys
import os
from pathlib import Path


# Configuration for different experiments
EXPERIMENTS = {
    "gradient_periodic": {
        "script": "gradient_periodic.py",
        "cycles": ["1", "3", "7", "15", "31", "63", "127"],
        "base_args": [
            "--max-iters", "512",
        ]
    },
    "gradient_linesearch": {
        "script": "gradient.py", 
        "cycles": ["linesearch"],  # Just one run for line search
        "base_args": [
            "--max-iters", "512",
        ]
    }
}


def run_experiment(solver_type, cycle, extra_args=None):
    """
    Run a single experiment with energy logging.
    
    Parameters:
    -----------
    solver_type : str
        Type of solver ("gradient_periodic" or "gradient_linesearch")
    cycle : str
        Cycle name (e.g., "3", "7", "linesearch")
    extra_args : list
        Additional command line arguments
    """
    
    config = EXPERIMENTS[solver_type]
    script = config["script"]
    base_args = config["base_args"].copy()
    
    if extra_args:
        base_args.extend(extra_args)
    
    # Generate energy log filename
    if solver_type == "gradient_linesearch":
        energy_log = f"energies_{solver_type}.csv"  # Just use solver type, no cycle name
    else:
        energy_log = f"energies_{solver_type}_{cycle}.csv"
    
    # Generate output filename  
    if solver_type == "gradient_linesearch":
        output_file = f"{solver_type}.pvd"
    else:
        output_file = f"{solver_type}_{cycle}.pvd"
    
    # Build command
    cmd = ["python3", script]
    cmd.extend(base_args)
    cmd.extend(["--energy-log", energy_log])
    cmd.extend(["--output", output_file])
    
    # Add cycle-specific arguments
    if solver_type == "gradient_periodic":
        cmd.extend(["--cycle-length", cycle])
    
    print(f"\n{'='*60}")
    print(f"Running {solver_type} with cycle {cycle}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Energy log: output/{energy_log}")
    print(f"{'='*60}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"✓ Completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with return code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"✗ Could not find script: {script}")
        print("Make sure you're in the correct directory.")
        return False


def run_all_experiments(solver_types=None, cycles=None, extra_args=None):
    """
    Run all experiments or a subset.
    
    Parameters:
    -----------
    solver_types : list or None
        List of solver types to run. If None, runs all.
    cycles : list or None  
        List of cycles to run. If None, runs all for each solver.
    extra_args : list
        Additional command line arguments for all runs
    """
    
    if solver_types is None:
        solver_types = list(EXPERIMENTS.keys())
    
    results = {}
    total_runs = 0
    successful_runs = 0
    
    for solver_type in solver_types:
        if solver_type not in EXPERIMENTS:
            print(f"Warning: Unknown solver type '{solver_type}', skipping.")
            continue
            
        config = EXPERIMENTS[solver_type]
        test_cycles = cycles if cycles is not None else config["cycles"]
        
        results[solver_type] = {}
        
        for cycle in test_cycles:
            total_runs += 1
            success = run_experiment(solver_type, cycle, extra_args)
            results[solver_type][cycle] = success
            if success:
                successful_runs += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {total_runs - successful_runs}")
    
    for solver_type, solver_results in results.items():
        print(f"\n{solver_type}:")
        for cycle, success in solver_results.items():
            status = "✓" if success else "✗"
            print(f"  {cycle}: {status}")
    
    # List generated files
    output_dir = Path("output")
    if output_dir.exists():
        csv_files = list(output_dir.glob("energies_*.csv"))
        if csv_files:
            print(f"\nGenerated energy logs:")
            for f in sorted(csv_files):
                print(f"  - {f}")
            
            print(f"\nTo plot results, run:")
            print(f"  python3 plot_convergence.py")
            print(f"  # or for specific files:")
            print(f"  python3 plot_convergence.py {' '.join(str(f) for f in csv_files[:3])}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run gradient descent experiments with energy logging")
    parser.add_argument("--solver", choices=list(EXPERIMENTS.keys()), action="append",
                      help="Solver type(s) to run. Can be specified multiple times. Default: all")
    parser.add_argument("--cycle", action="append", 
                      help="Specific cycle(s) to run. Can be specified multiple times. Default: all for each solver")
    parser.add_argument("--quick", action="store_true",
                      help="Run quick experiments (fewer iterations, coarser mesh)")
    parser.add_argument("--high-res", action="store_true", 
                      help="Run high resolution experiments (more iterations, finer mesh)")
    
    args = parser.parse_args()
    
    # Prepare extra arguments based on flags
    extra_args = []
    if args.quick:
        extra_args.extend(["--nx", "32", "--max-iters", "50"])
    elif args.high_res:
        extra_args.extend(["--nx", "128", "--max-iters", "500"])
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Run experiments
    run_all_experiments(
        solver_types=args.solver,
        cycles=args.cycle,
        extra_args=extra_args
    )


if __name__ == "__main__":
    main()