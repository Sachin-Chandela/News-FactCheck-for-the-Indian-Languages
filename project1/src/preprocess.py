import pandas as pd

def main():
    df1 = pd.read_csv("../data/facts_merged_final.csv")
    df2 = pd.read_csv("../data/factcheck_dataset_modified.csv")

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates(subset=["claim"])

    label_map = {
        "true": 0,
        "false": 1,
        "misleading": 2,
        "other": 3
    }

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(label_map)
    )

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df.to_csv("../data/final_dataset.csv", index=False)
    print("Saved: ../data/final_dataset.csv")
    print("Rows:", len(df))

if __name__ == "__main__":
    main()