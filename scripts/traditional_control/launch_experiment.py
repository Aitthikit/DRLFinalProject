import os
import sys
import runpy
import builtins
import time
from isaaclab.app import AppLauncher
import argparse

# ---- STEP 1: Add paths for import access ----
play_path = "/home/memekhos/DRLFinalProject/scripts/traditional_control/plaympc.py"
play_dir = os.path.dirname(play_path)

sys.path.insert(0, play_dir)

# ---- STEP 2: Modify DIPClabset.py configuration ----
# Simulate CLI arguments
import numpy as np

direc = 10
Np = 125
Nc = 20
Ws = [5.0, 10.0, 10.0, 1,1,1]
# Example changes
# ---- STEP 3: Monkey-patch play.py variables if needed ----
builtins._custom_test_name = f"MPCTune2_Pre{Np}"
builtins._custom_Np = Np
builtins._custom_Nc = Nc
builtins._custom_Ws = Ws

# ---- STEP 4: Inject CLI args and run play.py ----
sys.argv = [
    "plaympc.py",
    "--task=LAB",
    # "--headless"
]
# ---- STEP 5: Run the play.py script ----
runpy.run_path(play_path, run_name="__main__")
# time.sleep(15)
print(f"=== Finished Experiment {i+1} ===")
    
