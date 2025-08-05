import subprocess
import sys

if __name__ == "__main__":
    # Run pytest
    print("Running tests...")
    result = subprocess.run([sys.executable, "-m", "pytest"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        print("Tests failed.")
        sys.exit(result.returncode)
    print("Tests passed. Running tape simulator...")
    # Import and run the tape simulator
    from src.hardware.cassette_tape import CassetteTapeBackend
    # Example: instantiate and run a basic simulation
    tape = CassetteTapeBackend()
    print(f"Tape initialized with {tape.total_bits} bits.")
    # You can add more simulation logic here as needed
