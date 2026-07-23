from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "HUST"

pkl_files = sorted(DEFAULT_DATA_DIR.glob("*.pkl"))
rows = []

for i in pkl_files:
    with open(i, "rb") as f:
        data = pickle.load(f)

    cycle_data_name = "cycle_data"

    cycle_data = data[cycle_data_name]

    # print(type(cycle_data)) # list 
    # print(len(cycle_data)) # 1번 파일: 1504 

    cycle_data0 = cycle_data[0]
    # print(type(cycle_data0)) # dict
    # print(len(cycle_data0)) # 8

    for key, value in cycle_data0.items():
        print(key, value)
        print('------------------')
    '''
    cycle_number
    current_in_A
    voltage_in_V
    charge_capacity_in_Ah
    discharge_capacity_in_Ah
    time_in_s
    temperature_in_C
    internal_resistance_in_ohm
    '''
    print('------------')
    break