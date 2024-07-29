class User:
    def __init__(self, x, y, task=15, offload=0.1, hopeTime=1, relation=None, power=0.1, server=None):
        # task : 15-30
        self.x = x
        self.y = y
        self.task = task
        self.offload = offload
        self.hopeTime = hopeTime
        self.relation = relation
        self.power = power
        self.server = server

    def set_task(self, task):
        self.task = task

    def set_unload(self, offload):
        self.offload = offload

    def set_time(self, hopeTime):
        self.hopeTime = hopeTime

    def set_relation(self, relation):
        self.relation = relation
    
    def set_power(self, power):
        self.power = power

    def __str__(self):
        return f"User at ({self.x}, {self.y}) with task {self.task}, unload {self.unload}, time {self.time}, relation {self.relation}"
