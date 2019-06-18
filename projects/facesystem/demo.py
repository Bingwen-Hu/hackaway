import facesystem


def insert():
    x1 = 'xijinping/0.jpg'
    m1 = 'mayun/76.jpg'
    xinfo = {'name': '习近平', 'QQ': 12355, 'title': 'Principle'}
    minfo = {'name': '马云', 'QQ': 12356, 'title': 'Chief'}
    facesystem.face_register(m1, minfo)
    facesystem.face_register(x1, xinfo)


def recognize():
    x2 = 'xijinping/2.jpg'
    m2 = 'mayun/77.jpg'
    print(facesystem.face_recognize(m2))
    print(facesystem.face_recognize(x2))


def check_backup():
    print(facesystem.api.facedb.info)