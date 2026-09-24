from pathlib import Path
import os, sys, subprocess

OFFICIAL_REPO = "https://github.com/speckean/UPAR-Challenge-2027.git"

def prepare_official_repo(repo_dir="/content/UPAR-Challenge-2027"):
    repo = Path(repo_dir)
    if not repo.exists():
        subprocess.check_call(["git","clone","--depth","1",OFFICIAL_REPO,str(repo)])
    else:
        subprocess.run(["git","-C",str(repo),"pull","--ff-only"], check=False)

    sys.path.insert(0, str(repo))
    old = os.getcwd()
    os.chdir(repo)
    try:
        import download_datasets as dd
        data = repo / "data"
        data.mkdir(exist_ok=True)
        if not (data/"Market1501").exists():
            dd.prepare_market(data)
        else:
            print("[OK] Market1501 already present.")
        if not (data/"PA100k").exists() or not any((data/"PA100k").rglob("*.jpg")):
            dd.prepare_pa100k(data)
        else:
            print("[OK] PA100k already present.")
        if not (data/"PETA"/"images").exists() or not any((data/"PETA"/"images").glob("*")):
            dd.prepare_peta(data)
        else:
            print("[OK] PETA already present.")
    finally:
        os.chdir(old)

    train = repo/"data/annotations/task1/train/gt.csv"
    val = repo/"data/annotations/task1/val/gt.csv"
    if not train.exists() or not val.exists():
        raise FileNotFoundError("Official task1 annotation CSVs were not found after cloning.")
    print("Official UPAR 2027 data repository ready:", repo)
    return repo
