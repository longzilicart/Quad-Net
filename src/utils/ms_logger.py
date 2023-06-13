import logging
import os
from datetime import datetime
import time

def loadLogger(args):
    '''logger for DDP'''
    logger = logging.getLogger()
    #logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                 datefmt="%a-%b %d %H:%M:%S %Y")
    shandler = logging.StreamHandler()
    shandler.setFormatter(formatter)
    logger.addHandler(shandler)
    
    # if notsave, generate fhandler
    if not args.not_save and args.local_rank in [-1,0]: #rank in [-1,0]DDP专用
        if args.log_dir is not None:
            log_path = os.path.join(args.log_root, args.log_dir)
        else:
            log_path = os.path.join(args.log_root,
                                   time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
        if not os.path.exists(log_path):
            os.makedirs(log_path)        
        fhandler = logging.FileHandler(log_path + args.log_name,mode = 'a')
        fhandler.setLevel(logging.INFO)
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
    return logger