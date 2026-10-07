"""Fetch + validate + version Hillstrom, save the train/test split and write DATA_CARD.md."""
from uplift_alloc.config import ROOT
from uplift_alloc.data import data_card_markdown, prepare, save_split, validate

if __name__ == "__main__":
    df = prepare()
    split = save_split()
    (ROOT / "DATA_CARD.md").write_text(data_card_markdown(df, validate(df)), encoding="utf-8")
    print(f"{len(df):,} rows; train {len(split['train']):,} / test {len(split['test']):,}")
