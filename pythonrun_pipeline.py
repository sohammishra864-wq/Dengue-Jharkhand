
import subprocess
import sys


def run(script):
    print("\n" + "=" * 80)
    print(f"RUNNING: {script}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, script],
        check=True
    )

    print(f"✓ COMPLETED: {script}")


if __name__ == "__main__":

    run("premerge_qc.py")

    run("merge_extended_features.py")

    run("postmerge_qc.py")

    run("feature_engineering.py")

    run("engineered_qc.py")

    run("exploratory_analysis.py")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)