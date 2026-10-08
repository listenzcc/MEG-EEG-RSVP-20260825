#!/usr/bin/env python3

import argparse
import subprocess

parser = argparse.ArgumentParser(description='Git pull or push script')
parser.add_argument('action', choices=['pull', 'push'], help='Action to perform')


def main():
    args = parser.parse_args()
    print(args)
    try:
        result = subprocess.run(
            # ["git", "push"],
            ["git", args.action],
            capture_output=True,  # Capture both stdout and stderr
            text=True,           # Return output as string (not bytes)
            check=True           # Raise exception if command fails
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)

    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        main()

    except FileNotFoundError:
        print("Error: git command not found")


if __name__ == '__main__':
    main()
