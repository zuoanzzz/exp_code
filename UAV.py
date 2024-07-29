class UAV:
    def __init__(self, x, y, z, coverage=None, capacity=None, price=None, frequency=15, users=None, con_users=None):
        self.x = x
        self.y = y
        self.z = z
        self.coverage = coverage
        self.capacity = capacity
        self.price = price
        self.frequency = frequency
        self.users=users
        self.con_users = con_users
        

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
