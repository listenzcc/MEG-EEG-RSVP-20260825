#!/usr/bin/env python3

import subprocess


def main():
    try:
        result = subprocess.run(
            ["git", "push"],
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
