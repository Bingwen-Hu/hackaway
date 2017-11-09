# chapter 11 breakpoint
import pdb

def spam():
	print('The begin of spam')
	print('The end of spam')
	
if __name__ == '__main__':
	pdb.set_trace()
	spam()