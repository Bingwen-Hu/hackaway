# Tern
# domain: set of inputs
# co-domain: a set from which the function's output values are chosen.
# image: set of outputs, may smaller than co-domain

# this procedure is a real function in math.
def caesar(plaintext: str):
    code = []
    for char in plaintext:
        code_ = ord(char) + 3
        if char in "xyzXYZ":
            code_ = code_ - 26
        code_ = chr(code_) 
        code.append(code_)
    return "".join(code)