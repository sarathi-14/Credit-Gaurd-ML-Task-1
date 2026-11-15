import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "cs-training.csv": "https://raw.githubusercontent.com/timmoti/give_me_some_credit/master/cs-training.csv",
    "cs-test.csv": "https://raw.githubusercontent.com/timmoti/give_me_some_credit/master/cs-test.csv"
}

def download_datasets():
    print(f"Target data directory: {DATA_DIR}")
    for filename, url in FILES.items():
        dest = os.path.join(DATA_DIR, filename)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            print(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, dest)
            print(f"Saved {filename} ({os.path.getsize(dest) / 1024 / 1024:.2f} MB)")
        else:
            print(f"{filename} already exists ({os.path.getsize(dest) / 1024 / 1024:.2f} MB)")

    # Verify files
    train_path = os.path.join(DATA_DIR, "cs-training.csv")
    test_path = os.path.join(DATA_DIR, "cs-test.csv")
    
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path, nrows=5)
        print("\n--- cs-training.csv sample columns & shape check ---")
        print("Columns:", list(df_train.columns))
        print("Sample shape (first 5 rows):", df_train.shape)

    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path, nrows=5)
        print("\n--- cs-test.csv sample columns & shape check ---")
        print("Columns:", list(df_test.columns))
        print("Sample shape (first 5 rows):", df_test.shape)

if __name__ == "__main__":
    download_datasets()
