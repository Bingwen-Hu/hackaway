# chapter11 debug 
import logging

log_format = (
	'[%(relativeCreated)d % (levelname)s] '
	'%(pathname)s:%(lineno)d:%(funcName)s: %(message)s'
)

logging.basicConfig(level=logging.DEBUG, format=log_format)

def spam(a, b=123):
	return 'some spam'
	
spam(1)
spam(1, 456)
spam(b=1, a=456)

