import argparse
import os
import yaml
import re
from typing import Dict
import logging
from datetime import datetime
import sys

path_matcher = re.compile(r'\$\{([^}^{]+)\}')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def path_constructor(loader, node):
    ''' Extract the matched value, expand env variable, and replace the match '''
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end():]


yaml.add_implicit_resolver('!config', path_matcher)
yaml.add_constructor('!config', path_constructor)


def process_config_file(config_path: str) -> Dict:
    try:
        with open(config_path) as config_yaml:
            config = yaml.load(config_yaml, Loader=yaml.FullLoader)
        return config
    except:
        raise Exception(f"Unable to process config file at {config_path}!")
    
def setup_logging(config):
    '''
    configure and return logger object
    '''
    verbose = config["verbose"] if "verbose" in list(config.keys()) else False
    if not verbose:
        return None
    else:
        try:
            log_path = config["log_path"]
            logger = logging.getLogger("causal_inference")
            handler = logging.FileHandler(os.path.join(log_path,
                                                    'causal_inference_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.txt'),
                                           mode = 'a')
            logger.addHandler(handler)

            logger.setLevel(logging.DEBUG)
            
            return logger
        except Exception as e:
            print(f'Exception occurred in step setup_logging(): {e}')