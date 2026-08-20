"""
Run this locally (VS Code) to understand the dataset structure.
The actual training runs on Kaggle.
"""
from datasets import load_dataset
import pandas as pd


def preview_hinglish_conversations():
    print("Loading Hinglish-Everyday-Conversations-1M (first 20 rows)...")
    ds = load_dataset("Abhishekcr448/Hinglish-Everyday-Conversations-1M",
                       split="train", streaming=True)

    rows = []
    for i, row in enumerate(ds):
        if i >= 20:
            break
        rows.append(row)

    print(f"\nColumn names: {list(rows[0].keys())}")
    print(f"\nFirst 5 rows:")
    for r in rows[:5]:
        print(r)


def preview_hinglish_dataset():
    print("\nLoading Hinglish-Everyday-Conversations (first 20 rows)...")
    ds2 = load_dataset("findnitai/english-to-hinglish",
                        split="train", streaming=True)

    rows = []
    for i, row in enumerate(ds2):
        if i >= 20:
            break
        rows.append(row)

    print(f"\nColumn names: {list(rows[0].keys())}")
    print(f"\nFirst 5 rows:")
    for r in rows[:5]:
        print(r)


if __name__ == "__main__":
    preview_hinglish_conversations()
    preview_hinglish_dataset()