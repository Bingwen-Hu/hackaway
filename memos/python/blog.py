#!/usr/bin/python3
# helper to generate markdown-format for pelican blog generator.

# when to use?
# after I finish one blog, and copy it from hackaway to my Github Page

# Title: os 模块
# Date: 2019-10-10 11:03
# Modified: 2019-10-10 11:03
# Category: Python
# Tags: Python, 操作系统
# Authors: 默安
# Summary: Python系统操作模块os简介
# 
# what generated is
# Title: [fetch from first line]
# Date: [generate just in time]
# Modified: [same as date or generate just in time]
# Category: [the folder hold the blog]
# Tags: Category or pass by command line
# Authors: siriusdemon
# Summary: Empty right now
#
import os
import os.path as osp
from datetime import datetime

def blog_already(filename):
    with open(filename) as f:
        firstline = f.readline()
    
    if 'Title:' in firstline:
        return True
    return False


def create_pelican_head(filename):
    """for an raw blog"""
    with open(filename) as f:
        title = f.readline().lstrip("#").strip()
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    modify = date
    abspath = osp.abspath(filename)
    dirname = osp.dirname(abspath)
    folder = dirname.split(osp.sep)[-1]
    category = folder
    tags = category
    authors = "siriusdemon"
    summary = ""
    pelican_head = (
        f"Title: {title}\n"
        f"Date: {date}\n"
        f"Modified: {modify}\n"
        f"Category: {category}\n"
        f"Tags: {tags}\n"
        f"Authors: {authors}\n"
        f"Summary: {summary}\n"
    )
    return pelican_head


def add_pelican_head(filename, override=False):
    head = create_pelican_head(filename)
    with open(filename) as f:
        content = f.readlines()
        content[0] = head
    if not override:
        filename = f"new_{filename}"
    with open(filename, 'w') as f:
        f.writelines(content)
    
    print(f"Done! Save into {filename}")


def modify():
    """for an modified blog"""
    pass

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    already = blog_already(filename)
    if not already:
        add_pelican_head(filename, override=True)
    else:
        print("already blog!")