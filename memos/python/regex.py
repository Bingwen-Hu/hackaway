# python named group in regular expression
import unittest
import re
class Regex(unittest.TestCase):

    def test_group(self):
        string = '发布自 秒拍视频 http://t.cn/hahah'
        pattern = '(?P<source>.{6})http://[a-zA-Z./\?]+'
        res = re.search(pattern, string)
        source = res.group('source')
        self.assertIn('秒拍视频', source)
