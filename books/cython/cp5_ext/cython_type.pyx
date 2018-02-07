cdef class Particle_cy:
    """simple Particle extension type."""
    cdef readonly double mass      # public access, read only
    cdef double position           # private access
    cdef public double velocity    # public access, write and read
    
    def __init__(self, m, p, v):
        self.mass = m
        self.position = p
        self.velocity = v
    def get_nomentum(self):
        return self.mass * self.velocity

class Particle_py:
    def __init__(self, m, p, v):
        self.mass = m
        self.position = p
        self.velocity = v
    def get_nomentum(self):
        return self.mass * self.velocity
