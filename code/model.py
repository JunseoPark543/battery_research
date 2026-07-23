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
    print(len(cycle_data))

    cycle_data0 = cycle_data[0]
    
    for key, values in data.items():
        print(key)
        '''
        cell_id
        cycle_data
        form_factor
        anode_material
        cathode_material
        electrolyte_material
        nominal_capacity_in_Ah
        depth_of_charge
        depth_of_discharge
        already_spent_cycles
        max_voltage_limit_in_V
        min_voltage_limit_in_V
        max_current_limit_in_A
        min_current_limit_in_A
        reference
        description
        charge_protocol
        discharge_protocol
        SOC_interval
        '''

    print('------------')
    break