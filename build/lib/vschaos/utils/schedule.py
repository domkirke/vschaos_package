
class Scheduled(float):

    def __new__(cls, value, *args, **kwargs):
        return super().__new__(cls, value)

    def __init__(self, value=0, epoch=0, period=None):
        if issubclass(type(value), (float, int)):
            self.value = value
            self.epoch = epoch
        elif issubclass(type(value), Scheduled):
            self.value = value.value
            self.epoch = value.epoch
        self.update(epoch)

    def update(self, epoch, period=None):
        self.epoch = epoch
        self.period = None
        self.value = self.get_value(epoch)

    def get_value(self, epoch=None):
        return self.value

    def __str__(self):
        return str(float(self))

    def __repr__(self):
        return "Scheduled()"

    def __hash__(self):
        return self.get_value(self.value).__hash__()

    def __repr__(self):
        return "caca"

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __mul__(self, other):
        return float(self)*other

    def __add__(self, other):
        return float(self) + other

    def __call__(self, *args, **kwargs):
        print('coucou')

    def __lt__(self, other):
        return float(self) < other

    def __gt__(self, other):
        return float(self) > other

    def __le__(self, other):
        return float(self) <= other

    def __ge__(self, other):
        return float(self) >= other

    def __eq__(self, other):
        return float(self) == other

    def __ne__(self, other):
        return float(self) != other


class LinearSchedule(Scheduled):
    def __new__(cls, value=0.0, *args, **kwargs):
        return super().__new__(cls, value)

    def __repr__(self):
        return "LinearSchedule(range=%s, values=%s, current=%s)"%(self.range, self.values, float(self))

    def __init__(self, *args, range=[0, 100], values=[0.0, 1.0], **kwargs):
        self.range = range
        self.values = values
        super(LinearSchedule, self).__init__(*args, **kwargs)

    def get_value(self, epoch=None):
        epoch = epoch or self.epoch
        if self.range[1] == self.range[0]:
            if epoch < self.range[0]:
                return self.values[0]
            else:
                return self.values[1]
        a = (self.values[1] - self.values[0])/(self.range[1] - self.range[0])
        b = self.values[0] - a*self.range[0]
        return a*epoch + b


class Warmup(LinearSchedule):

    def __repr__(self):
        return "Warmup(range=%s, values=%s, current=%s)"%(self.range, self.values, float(self))

    def get_value(self, epoch=None):
        return max(self.values[0], min(self.values[1], super(Warmup, self).get_value(epoch)))


class Threshold(Scheduled):

    def __repr__(self):
        return "Threshold(value=%s, threshold=%s)"%(self.threshold, float(self))

    def __new__(cls, value=1.0, *args, **kwargs):
        return super().__new__(cls, value)

    def __init__(self, value=1.0, *args, threshold=100, **kwargs):
        self.threshold = threshold
        self.ref_value = value
        super(Threshold, self).__init__(0., *args, **kwargs)

    def get_value(self, epoch=None):
        epoch = epoch or self.epoch
        if epoch < self.threshold:
            return 0.
        else:
            return self.ref_value
