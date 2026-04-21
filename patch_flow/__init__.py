import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)