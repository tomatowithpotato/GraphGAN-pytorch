
class graph:
    class node:
        def __init__(self, id, name="default"):
            self.id = id
            self.name = name
            self.neighbors = []

    def __init__(self, id, name="default"):
        self.id = id
        self.name = name
        self.G = {}
