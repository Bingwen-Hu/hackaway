# -*- coding: utf-8 -*-
""" argparse test

@author: Mory
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', help='mysql host')
parser.add_argument('--user', help='mysql username')
parser.add_argument('--passwd', help='mysql password')
parser.add_argument('--port', help='mysql port')
parser.add_argument('--receiver', help='receiver username')
parser.add_argument('--sender', help='sender username')
parser.add_argument('--sender_pass', help='sender password')


args = parser.parse_args()
print(parser.format_help())
print(args.host)
print(args.passwd)
print(args.sender)