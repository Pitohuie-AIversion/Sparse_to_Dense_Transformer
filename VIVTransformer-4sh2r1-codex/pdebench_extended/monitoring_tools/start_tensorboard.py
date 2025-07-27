import os
import subprocess
from pathlib import Path

def main():
    """Starts a TensorBoard server to monitor training results."""
    project_root = Path(__file__).resolve().parent
    log_dir = project_root / "attention_results"
    
    print(f"Starting TensorBoard with logdir: {log_dir}")
    
    try:
        # Use subprocess.run to start TensorBoard
        # This will block until TensorBoard is manually stopped (e.g., with Ctrl+C)
        subprocess.run(["tensorboard", "--logdir", str(log_dir)], check=True)
    except FileNotFoundError:
        print("Error: 'tensorboard' command not found.")
        print("Please make sure TensorBoard is installed and in your PATH.")
        print("You can install it with: pip install tensorboard")
    except subprocess.CalledProcessError as e:
        print(f"TensorBoard exited with an error: {e}")
    except KeyboardInterrupt:
        print("\nTensorBoard server stopped.")

if __name__ == "__main__":
    main()