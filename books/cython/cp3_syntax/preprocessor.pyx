DEF E = 2.71828
DEF PI = 3.1415


print(E)
print(PI)


# predefine macro
IF UNAME_SYSNAME == "Windows":
    print("running on windows")
ELIF UNAME_SYSNAME == "Linux":
    print("on Linux")
ELSE:
    print("Maybe OS X")
