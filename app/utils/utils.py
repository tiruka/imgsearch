import glob
import logging
import os

from settings import (
    MODEL_DIR,
    CHECK_POINT,
    LOG_DIR
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_glob(pathname, recursive=True):
    for p in glob.glob(pathname, recursive=recursive):
        if os.path.isfile(p):
            os.remove(p)


def clear_under_vars():
    logger.info('Clear models under {}'.format(MODEL_DIR))
    remove_glob(os.path.join(MODEL_DIR, '*.h5'), recursive=False)
    remove_glob(os.path.join(MODEL_DIR, '*.pkl'), recursive=False)
    remove_glob(os.path.join(MODEL_DIR, '*.json'), recursive=False)
    logger.info('Clear checkpoints under {}'.format(CHECK_POINT))
    remove_glob(os.path.join(CHECK_POINT, '*.h5'), recursive=False)
    logger.info('Clear logs under {}'.format(LOG_DIR))
    remove_glob(os.path.join(LOG_DIR, '*.csv'), recursive=False)