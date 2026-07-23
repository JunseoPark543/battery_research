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

    
    x = cycle_data0["time_in_s"]
    y = cycle_data0["current_in_A"]
    plt.plot(x, y)
    
    plt.xlabel("time_in_s")
    plt.ylabel("current_in_A")
    plt.show()
    '''
    cycle_number                    int
    current_in_A                    list 
    voltage_in_V                    list
    charge_capacity_in_Ah           list
    discharge_capacity_in_Ah        list
    time_in_s                       list 
    temperature_in_C                None 
    internal_resistance_in_ohm      None 
    '''
    break