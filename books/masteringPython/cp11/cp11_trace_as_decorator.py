# exercise trace as decorator
import sys
import trace as trace_module
import contextlib



def trace(f, *args, **kwargs):
	def trace_func(*args, **kwargs):
		tracer = trace_module.Trace(
			False, trace=True, timing=True)
		sys.settrace(tracer.globaltrace)
		yield tracer 
		f(*args, **kwargs)
		sys.settrace(None)
	
		result = tracer.results()
		result.write_results(show_missing=False, summary=True)
	return trace_func
	

def eggs_generator():
	yield 'eggs'
	yield 'EGGS!'

@trace
def spam_generator():
	yield 'spam'
	yield 'spam!'
	yield 'SPAM!'
	
generator = spam_generator()
print(next(generator))
print(next(generator))
	
generator = eggs_generator()
print(next(generator))

# wait to correct
	