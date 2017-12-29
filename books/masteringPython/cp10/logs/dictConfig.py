import logging
from logging import config

config.dictConfig({
    'version': 1,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s',
        },
    },
    'handlers': {
        'file': {
            'filename': 'debug.log',
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'formatter': 'standard',
        },
        'stream': {
            'level': 'WARNING',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },
    'loggers': {
        "": {
            'handlers': ['file', 'stream'],
            'level': 'DEBUG',
        },
    },
})

logging.debug('debug')
logging.info('info')
logging.warning('warning')
logging.error('error')
logging.critical('critical')
