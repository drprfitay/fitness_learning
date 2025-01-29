#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 14:17:08 2025

@author: itayta
"""

import sys, os
from constants import *

original_sys_path = ""
def fix_esm_path():
    global original_sys_path
    # Specify the module name and path
    module_name = "esm"
    module_path = ITAYFOLD_PATH 
    # Store the original sys.path
    original_sys_path = sys.path.copy()
    # Temporarily add the local directory to sys.path
    sys.path.insert(0, os.path.abspath(module_path))
    # hack
    for mdl in [k for k,v in sys.modules.items() if module_name in k]:
        del sys.modules[mdl]

