# thanks for Lisp
from random import choice, randint


def complete_answer(host, guest, flag=0):
    """
    flag=0 代表生成的句子应该是host正向
    反之，生成的是host负向
    """
    rteam, team = teams(host, guest)
    # if flag == 0:   # 要产生host正向或者guest负向
    #     if rteam == 0: # 生成了host，那么词汇应是正向的
    #         v = adv_verb()
    #     else:       # 否则词汇是负向的
    #         v = iadv_verb()
    # else:           # 要产生host负向或者guest正向
    #     if rteam == 0: # 生成了host，那么词汇应是负向的
    #         v = iadv_verb()
    #     else:
    #         v = adv_verb()
    if flag == rteam:
        v = adv_verb()
    else:
        v = iadv_verb()
    answer = team + v
    answer = pre_subject(answer)
    return answer

def teams(host, guest):
    r = randint(0, 1)
    if r == 0:
        return r, host
    return r, guest

def adv_verb():
    ad = choice([
        "会赢", "必定赢", "可能会赢", "很可能赢", "肯定会赢", "肯定赢",
        "是不可能会输的", "应该能赢", "会取胜",
    ])
    return ad

def iadv_verb():
    return choice(["不可能赢", "是赢不了的", "估计会输", "赢不了"])

def pre_subject(answer):
    pre = choice(['我认为', "", "我觉得", "你要问我的话，我认为", "我认为", "", "", "", "", "", ""])
    if pre:
        answer = f"{pre}{answer}"
    return answer


def vote_answer(flag=0):
    if flag == 0:
        msg = "我赞成小新的观点"
    msg = "我不赞成小新的观点"
    return msg

def analysis_stand(answer, host, guest):
    host_in = host in answer
    pos_in = "不" not in answer
    win_in = "赢" in answer or "胜" in answer
    
    if host_in:
        if pos_in == win_in:
            flag = 0
        else:
            flag = 1
    else:
        if pos_in == win_in:
            flag = 1
        else:
            flag = 0
    return flag

    # host_in + pos_in + win_in ==> 0  主赢
    # host_in + !pos_in + win_in ==> 1 主不赢
    # host_in + pos_in + !win_in ==> 1 主不赢
    # host_in + !pos_in + !win_in ==> 0 主不输
    # !host_in + pos_in + win_in ==> 1 客赢
    # !host_in + !pos_in + win_in ==> 0 客不赢
    # !host_in + !pos_in + !win_in ==> 1 客不输
    # !host_in + pos_in + !win_in ==> 0 客输