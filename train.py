import argparse
import subprocess
import os
import glob
import random

def collect_yaml_files(paths):
    config_files = []

    for p in paths:
        if os.path.isfile(p) and p.endswith(('.yaml', '.yml')):
            config_files.append(os.path.abspath(p))
        elif os.path.isdir(p):
            yamls = glob.glob(os.path.join(p, '*.yaml')) + glob.glob(os.path.join(p, '*.yml'))
            yamls = [os.path.abspath(f) for f in yamls]
            config_files.extend(sorted(yamls))  # sort files inside the dir
        else:
            print(f"Warning: {p} is not a valid YAML file or directory. Skipping.")
    
    return sorted(config_files)  # overall sorting across all inputs


def main():
    parser = argparse.ArgumentParser(description='Train CALM with multiple config files or directories.')
    parser.add_argument('--configs', type=str, nargs='+', required=True,
                        help='YAML config file paths or directories containing YAML files')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the order of config execution')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of times to repeat running the config files')
    args = parser.parse_args()

    config_files = collect_yaml_files(args.configs)

    if args.shuffle:
        random.shuffle(config_files)

    if not config_files:
        print("No valid YAML config files found. Exiting.")
        return

    print("\nConfigs to be run (in order):", flush=True)
    for i, config_path in enumerate(config_files, 1):
        print(f"{i:2d}. {config_path}", flush=True)

    print(f"\nStarting training runs: {args.repeat} repetition(s)\n", flush=True)

    for rep in range(1, args.repeat + 1):
        print(f"=== Repetition {rep}/{args.repeat} ===", flush=True)
        for config_path in config_files:
            print(f"[Running main.py with config: {config_path}]", flush=True)
            subprocess.run(
                ['python', 'main.py', '--config', config_path],
                check=True
            )

if __name__ == '__main__':
    main()
