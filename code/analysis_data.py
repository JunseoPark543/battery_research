
from pathlib import Path
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "HUST"

pkl_files = sorted(DEFAULT_DATA_DIR.glob("*.pkl"))

for pkl_file in pkl_files:
    with pkl_file.open("rb") as f:
        data = pickle.load(f)

    exclude_key = "cycle_data"

    filtered_data = {
        key: value
        for key, value in data.items()
        if key != exclude_key
    }
    print(filtered_data)
'''
    'cell_id': 'HUST_9-8', 
    'form_factor': 'cylindrical_18650', 
    'anode_material': 'graphite', 
    'cathode_material': 'LFP', 
    'electrolyte_material': None, 
    'nominal_capacity_in_Ah': 1.1, 
    'depth_of_charge': 1.0, 
    'depth_of_discharge': 1.0, 
    'already_spent_cycles': 0, 
    'max_voltage_limit_in_V': 3.6, 
    'min_voltage_limit_in_V': 2.0, 
    'max_current_limit_in_A': None, 
    'min_current_limit_in_A': None, 
    'reference': None, 
    'description': None, 
    'charge_protocol': [
        {'rate_in_C': 5.0, 
        'current_in_A': None, 
        'voltage_in_V': None, 
        'power_in_W': None, 
        'start_voltage_in_V': None, 
        'start_soc': 0.0, 
        'end_voltage_in_V': None, 
        'end_soc': 0.8}, 
        {'rate_in_C': 1.0, 
        'current_in_A': None, 
        'voltage_in_V': None, 
        'power_in_W': None, '
        start_voltage_in_V': None, 
        'start_soc': 0.8, 
        'end_voltage_in_V': 3.6, 
        'end_soc': None}, 
        {'rate_in_C': None, 
        'current_in_A': None, 
        'voltage_in_V': 3.6, 
        'power_in_W': None, 
        'start_voltage_in_V': 3.6, 
        'start_soc': None, 
        'end_voltage_in_V': None, 
        'end_soc': 1.0}], 
    'discharge_protocol': [{'rate_in_C': 3.0, 'current_in_A': None, 'voltage_in_V': None, 'power_in_W': None, 'start_voltage_in_V': None, 'start_soc': 1.0, 'end_voltage_in_V': None, 'end_soc': 0.6}, {'rate_in_C': 5.0, 'current_in_A': None, 'voltage_in_V': None, 'power_in_W': None, 'start_voltage_in_V': None, 'start_soc': 0.6, 'end_voltage_in_V': None, 'end_soc': 0.4}, {'rate_in_C': 2.0, 'current_in_A': None, 'voltage_in_V': None, 'power_in_W': None, 'start_voltage_in_V': None, 'start_soc': 0.4, 'end_voltage_in_V': None, 'end_soc': 0.2}, {'rate_in_C': 1.0, 'current_in_A': None, 'voltage_in_V': None, 'power_in_W': None, 'start_voltage_in_V': None, 'start_soc': 0.2, 'end_voltage_in_V': 2.0, 'end_soc': None}], 
    'SOC_interval': [0, 1]}
'''