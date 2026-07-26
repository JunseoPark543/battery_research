import pickle

file_path = "./battery_research/data/HUST/HUST_1-2.pkl"
import pickle
import numpy as np
import pandas as pd

with open(file_path, "rb") as f:
    data = pickle.load(f)

cycle_data = data["cycle_data"]

print("cycle_data type:", type(cycle_data))

try:
    print("cycle_data length:", len(cycle_data))
except TypeError:
    print("cycle_data length: 확인 불가")


# cycle_data가 dictionary인 경우
if isinstance(cycle_data, dict):
    print("\ncycle_data keys:")
    print(list(cycle_data.keys())[:10])

    first_key = next(iter(cycle_data))
    first_cycle = cycle_data[first_key]

    print("\nfirst cycle key:", first_key)
    print("first cycle type:", type(first_cycle))

# cycle_data가 list 또는 tuple인 경우
elif isinstance(cycle_data, (list, tuple)):
    first_cycle = cycle_data[0]

    print("\nfirst cycle index: 0")
    print("first cycle type:", type(first_cycle))

# cycle_data가 ndarray인 경우
elif isinstance(cycle_data, np.ndarray):
    print("cycle_data shape:", cycle_data.shape)
    first_cycle = cycle_data[0]

    print("\nfirst cycle type:", type(first_cycle))
    print("first cycle shape:", np.asarray(first_cycle).shape)

# DataFrame인 경우
elif isinstance(cycle_data, pd.DataFrame):
    print("cycle_data shape:", cycle_data.shape)
    print("columns:", cycle_data.columns.tolist())
    print(cycle_data.head())
    first_cycle = None

else:
    first_cycle = None
    print("cycle_data:", cycle_data)


# 첫 번째 cycle의 내부 구조 확인
if first_cycle is not None:

    if isinstance(first_cycle, dict):
        print("\nfirst cycle keys:")
        print(first_cycle.keys())

        for key, value in first_cycle.items():
            print(f"\n[{key}]")
            print("type:", type(value))

            if hasattr(value, "shape"):
                print("shape:", value.shape)

            try:
                print("length:", len(value))
            except TypeError:
                pass

            if isinstance(value, (list, tuple, np.ndarray)):
                array = np.asarray(value)
                print("first values:", array.reshape(-1)[:5])

            elif isinstance(value, pd.DataFrame):
                print("columns:", value.columns.tolist())
                print(value.head())

            else:
                print("value:", value)

    elif isinstance(first_cycle, pd.DataFrame):
        print("\nfirst cycle shape:", first_cycle.shape)
        print("first cycle columns:", first_cycle.columns.tolist())
        print(first_cycle.head())

    elif isinstance(first_cycle, np.ndarray):
        print("\nfirst cycle shape:", first_cycle.shape)
        print("first cycle values:")
        print(first_cycle[:5])

    else:
        print("\nfirst cycle:")
        print(first_cycle)