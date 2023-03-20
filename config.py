#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Collection of functions to load the configuration file
"""
#
# Authors:  Felix MICHAUD   <felixmichaudlnhrdt@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#           Joachim POUTARAUD <joachipo@imv.uio.no>

import sys
import os
import yaml

RANDOM_SEED = 1979  # Fix the random seed to be able to repeat the results

PARAMS_MODEL = {
    "DEVICE": "cpu",
    "BATCH_SIZE": 256,
    "N_WORKERS": 0,
    "N_WAY": 5,
    "N_SHOT": 5,
    "N_QUERY": 1,
    "N_TASKS": 100,
    "N_EPOCHS": 200,
    "EARLY_STOP_THRESH": 5,
    "SCHEDULER_FACTOR": 0.1,
    "SCHEDULER_PATIENCE": 5,
    "LEARNING_RATE": 0.0001,
    "TRAINING": "classical",
    "PRETRAINED": True,
    "HPSS": True,
    "REMOVE_BG": True,
    "MODEL": "prototypical"
}

""" ===========================================================================

                    Private functions 

============================================================================"""
def _fun_call_by_name(val):
    if '.' in val:
        module_name, fun_name = val.rsplit('.', 1)
        # Restrict which modules may be loaded here to avoid safety issue
        # Put the name of the module
        assert module_name.startswith('bambird')
    else:
        module_name = '__main__'
        fun_name = val
    try:
        __import__(module_name)
    except ImportError :
        raise ("Can''t import {} while constructing a Python object".format(val))
    module = sys.modules[module_name]
    fun = getattr(module, fun_name)
    return fun

def _fun_constructor(loader, node):
    val = loader.construct_scalar(node)
    print(val)
    return _fun_call_by_name(val)

def _get_loader():
    """Add constructors to PyYAML loader."""
    loader = yaml.SafeLoader
    loader.add_constructor("!FUNC", _fun_constructor)
    return loader

""" ===========================================================================

                    Public function 

============================================================================"""

def load_config(fullfilename = None):
    """
    Load the configuration file to set all the parameters of bambird

    Parameters
    ----------
    fullfilename : string, optional
        Path to the configuration file.
        if no valid configuration file is given, the parameters are set to the
        default values.

    Returns
    -------
    PARAMS : dictionary
        Dictionary with all the parameters that are required for the bambird's
        functions
    """    
    global PARAMS_MODEL
    
    if os.path.isfile(str(fullfilename)): 
        with open(fullfilename) as f:
            PARAMS = yaml.load(f, Loader=_get_loader())
            PARAMS_MODEL = PARAMS['PARAMS_MODEL']
    else :
        print("The config file {} could not be loaded. Default parameters are loaded".format(fullfilename))
        
    return PARAMS

def get_config() :
    PARAMS = {
        'RANDOM_SEED' : RANDOM_SEED,
        'PARAMS_MODEL' : PARAMS_MODEL
        }
    return PARAMS


