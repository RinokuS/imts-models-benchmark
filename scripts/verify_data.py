"""Verify that benchmark datasets are present and structurally correct.

Run before starting any benchmark experiment:
    .bench_venv/bin/python scripts/verify_data.py

Exit codes: 0 = all OK, 1 = one or more datasets missing or malformed.
"""

import sys
from pathlib import Path

# Allow running from benchmark/ root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OK = "\033[92m✅\033[0m"
FAIL = "\033[91m❌\033[0m"
WARN = "\033[93m⚠️ \033[0m"


def check_p12(data_root: Path) -> bool:
    p12_dir = data_root / "physionet2012"
    print(f"\n{'='*60}")
    print("PhysioNet Challenge 2012")
    print(f"  Expected: {p12_dir}")

    if not p12_dir.exists():
        print(f"  {FAIL} Directory not found.")
        _p12_instructions(p12_dir)
        return False

    ok = True
    for subset in ["set-a", "set-b"]:
        subset_dir = p12_dir / subset
        if not subset_dir.exists():
            print(f"  {FAIL} {subset}/ not found in {p12_dir}")
            ok = False
        else:
            txt_files = list(subset_dir.glob("*.txt"))
            print(f"  {OK} {subset}/  ({len(txt_files)} patient files)")
            if len(txt_files) < 100:
                print(f"  {WARN} Only {len(txt_files)} files — expected ~4000")
                ok = False

    for outcomes_file in ["Outcomes-a.txt", "Outcomes-b.txt"]:
        path = p12_dir / outcomes_file
        if not path.exists():
            print(f"  {FAIL} {outcomes_file} not found")
            ok = False
        else:
            lines = path.read_text().strip().split("\n")
            print(f"  {OK} {outcomes_file}  ({len(lines)-1} records)")

    if ok:
        print(f"  {OK} PhysioNet P12 — OK")
    else:
        _p12_instructions(p12_dir)
    return ok


def check_p19(data_root: Path) -> bool:
    p19_dir = data_root / "physionet2019"
    print(f"\n{'='*60}")
    print("PhysioNet Challenge 2019")
    print(f"  Expected: {p19_dir / 'training'}")

    training_dir = p19_dir / "training"
    if not training_dir.exists():
        print(f"  {FAIL} Directory not found.")
        _p19_instructions(p19_dir)
        return False

    # Support both flat layout (training/*.psv) and nested (training/training_set*/*.psv)
    psv_files = list(training_dir.glob("*.psv"))
    if not psv_files:
        psv_files = list(training_dir.glob("**/*.psv"))
    n = len(psv_files)
    if n < 100:
        print(f"  {FAIL} Only {n} .psv files found — expected ~38,803")
        _p19_instructions(p19_dir)
        return False

    # Report the layout found
    subdirs = sorted({f.parent.name for f in psv_files if f.parent != training_dir})
    if subdirs:
        for sd in subdirs:
            cnt = sum(1 for f in psv_files if f.parent.name == sd)
            print(f"  {OK} training/{sd}/  ({cnt} .psv files)")
    else:
        print(f"  {OK} training/  ({n} .psv patient files)")
    print(f"  Total: {n} patient files")
    sample = psv_files[0]
    header = sample.read_text().split("\n")[0]
    if "|" not in header:
        print(f"  {FAIL} File format mismatch — expected PSV (pipe-separated), got: {header[:60]}")
        return False

    print(f"  {OK} PhysioNet P19 — OK")
    return True


def check_synthetic() -> bool:
    print(f"\n{'='*60}")
    print("Synthetic dataset")
    print("  Auto-generated — no files needed.")
    try:
        from data.synthetic import SyntheticDataset
        ds = SyntheticDataset(n_samples=10, seq_len=50, n_channels=5)
        t, v, m, y = ds[0]
        print(f"  {OK} Generated: times={tuple(t.shape)}, values={tuple(v.shape)}, label={y.item()}")
        return True
    except Exception as e:
        print(f"  {FAIL} Import error: {e}")
        return False


def _p12_instructions(p12_dir: Path):
    print(f"""
  HOW TO FIX:
  1. Register at https://physionet.org/register/
  2. Accept the DUA at https://physionet.org/content/challenge-2012/1.0.0/
  3. Download:
       wget --user=<USER> --password=<PASS> \\
         https://physionet.org/files/challenge-2012/1.0.0/set-a.zip \\
         https://physionet.org/files/challenge-2012/1.0.0/set-b.zip \\
         https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt \\
         https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt
  4. Extract:
       mkdir -p {p12_dir}
       unzip set-a.zip -d {p12_dir}/
       unzip set-b.zip -d {p12_dir}/
       mv Outcomes-a.txt Outcomes-b.txt {p12_dir}/
  See DATASETS.md for full instructions.
""")


def _p19_instructions(p19_dir: Path):
    print(f"""
  HOW TO FIX:
  1. Register at https://physionet.org/register/
  2. Accept the DUA at https://physionet.org/content/challenge-2019/1.0.0/
  3. Download:
       wget --user=<USER> --password=<PASS> \\
         https://physionet.org/files/challenge-2019/1.0.0/training/training_setA.zip \\
         https://physionet.org/files/challenge-2019/1.0.0/training/training_setB.zip
  4. Extract (merge both sets into training/):
       mkdir -p {p19_dir}/training
       unzip training_setA.zip -d {p19_dir}/
       unzip training_setB.zip -d /tmp/setB/
       mv /tmp/setB/training/*.psv {p19_dir}/training/
  See DATASETS.md for full instructions.
""")


def main():
    data_root = PROJECT_ROOT / "data" / "raw"

    print(f"Benchmark data root: {data_root}")
    print("Checking datasets...\n")

    results = {
        "synthetic": check_synthetic(),
        "p12": check_p12(data_root),
        "p19": check_p19(data_root),
    }

    print(f"\n{'='*60}")
    print("SUMMARY")
    all_ok = True
    for name, status in results.items():
        icon = OK if status else FAIL
        print(f"  {icon}  {name}")
        if not status:
            all_ok = False

    if all_ok:
        print("\nAll datasets ready. You can now run the benchmark.")
        print("  .bench_venv/bin/python scripts/run_all.py")
    else:
        missing = [n for n, s in results.items() if not s]
        if set(missing) == {"p12", "p19"}:
            print("\nPhysioNet data not downloaded yet.")
            print("Synthetic dataset is available — you can test models now:")
            print("  .bench_venv/bin/python scripts/train.py --model dlinear --dataset synthetic --epochs 5")
        else:
            print("\nFix the issues above and re-run this script.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
