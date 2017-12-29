# recommend logger name style

import logging

logger = logging.getLogger(__name__)


class Spam(object):
    def __init__(self, count):
        self.logger = logger.getChild(self.__class__.__name__)
