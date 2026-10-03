"""CLI helper script to launch the interactive Streamlit dashboard."""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    app_path = root / "dashboard" / "app.py"
    print(f"Launching Streamlit dashboard: {app_path}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")


if __name__ == "__main__":
    main()
