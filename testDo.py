class UAV:
    def __init__(self, x, y, z, coverage=None, capacity=None, price=None, frequency=15, users=None, con_users=None):
        self.x = x
        self.y = y
        self.z = z
        self.coverage = coverage
        self.capacity = capacity
        self.price = price
        self.frequency = frequency
        self.users = users if users is not None else []
        self.con_users = con_users if con_users is not None else []

    def set_coverage(self, coverage):
        self.coverage = coverage

    def set_capacity(self, capacity):
        self.capacity = capacity

    def set_price(self, price):
        self.price = price

    def set_frequency(self, frequency):
        self.frequency = frequency

    def set_users(self, users):
        self.users = users

    def __str__(self):
        return f"UAV at ({self.x}, {self.y}, {self.z}) with coverage {self.coverage}, capacity {self.capacity}, price {self.price} and frequency {self.frequency}"


class User:
    def __init__(self, x, y, task=15, offload=0.1, hopeTime=1, relation=None, power=0.1, server=None):
        self.x = x
        self.y = y
        self.task = task
        self.offload = offload
        self.hopeTime = hopeTime
        self.relation = relation if relation is not None else []
        self.power = power
        self.server = server

    def set_task(self, task):
        self.task = task

    def set_offload(self, offload):
        self.offload = offload

    def set_hopeTime(self, hopeTime):
        self.hopeTime = hopeTime

    def set_relation(self, relation):
        self.relation = relation

    def set_power(self, power):
        self.power = power

    def __str__(self):
        return f"User at ({self.x}, {self.y}) with task {self.task}, offload {self.offload}, time {self.hopeTime}, relation {self.relation}"
