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

def get_category(filename):
    abspath = osp.abspath(filename)
    dirname = osp.dirname(abspath)
    folder = dirname.split(osp.sep)[-1]
    category = folder
    return category

def create_pelican_head(filename):
    """for an raw blog"""
    with open(filename) as f:
        title = f.readline().lstrip("#").strip()
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    modify = date
    abspath = osp.abspath(filename)
    category = abspath.split(osp.sep)[4]
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


def modify(filename):
    """for an modified blog"""
    modify = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(filename) as f:
        content = f.readlines()
    # the 3-th line is modified time
    content[2] = f"Modified: {modify}\n"
    # update category
    category = get_category(filename)
    content[3] = f"Category: {category}\n"
    with open(filename, 'w') as f:
        f.writelines(content)
    print("modify blog {}".format(filename))



def create_footer(filename):
    abspath = osp.abspath(filename)
    # /home/mory/hackaway/{category}/xx/
    category = abspath.split(osp.sep)[4]
    dirname = osp.dirname(abspath)
    position = dirname[abspath.index(category):]
    github_url = osp.join("https://github.com/siriusdemon/hackaway/tree/master/",
                            position)

    foot = (
        "\n"
        "### Next\n"
        f"+ 所有的代码都可以在[Github]({github_url})获取。\n"
        "+ 关注我的[Github Page](https://siriusdemon.github.io/)查看更新。\n"
        "+ 也可以关注公众号可食用代码。\n"
        "\n"
        "![wechat](./images/wechat.jpg)\n"
        "\n"
        "### Wishes\n"
        "愿所有见过，听说过，忆念以及使用这篇文章的人，都能够获得暂时的快乐与永久不变的快乐。"
    )
    return category, foot


def generate(filename, head, footer, target_dir, target_name):
    with open(filename) as f:
        content = f.readlines()
    content[0] = head
    if content[-1] != '\n':
        content.append("\n")
    content.append(footer)
    target_path = osp.join(target_dir, target_name)
    with open(target_path, 'w', encoding='utf-8') as f:
        f.writelines(content)
    print(f"write into {target_path}... Done!")

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        target_name = sys.argv[2]
    else:
        target_name = filename
    already = blog_already(filename)

    if not already:
        target_dir = "/home/mory/hackaway/projects/github/content"
        head = create_pelican_head(filename)
        category, footer = create_footer(filename)
        target_dir = osp.join(target_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        generate(filename, head, footer, target_dir, target_name)
        # copy graphs
        graphs_dir = osp.join(os.getcwd(), 'graphs')
        if osp.exists(graphs_dir):
            os.makedirs(osp.join(target_dir, "graphs"), exist_ok=True)
            os.system(f"cp {graphs_dir}/* {target_dir}/graphs/")
            print("copy graphs ... Done!")
    else:
        modify(filename)