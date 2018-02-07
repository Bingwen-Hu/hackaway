cdef class Particle_cy:
    """simple Particle extension type."""
    cdef double mass, position, velocity
    
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
