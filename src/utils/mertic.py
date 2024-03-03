class Metrics:
    main_key: str
    downward: bool

    def __init__(self, *args):
        assert len(args) % 2 == 0, "Missing key or value, please make sure input is paired."
        self.metrics = {}
        self.num_samples = 0

        for kv in zip(args[0::2], args[1::2]):
            self.metrics[kv[0]] = kv[1]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __getitem__(self, item):
        return self.metrics[item]

    def __len__(self):
        return self.num_samples

    def add(self, key, value, *args):
        self.metrics[key] += value
        self.num_samples += 1
        assert len(args) % 2 == 0, "Missing key or value, please make sure input is paired."
        for kv in zip(args[0::2], args[1::2]):
            self.metrics[kv[0]] += kv[1]

    def reset(self):
        for k in self.metrics:
            self.metrics[k] = 0
            self.num_samples = 0

    @property
    def main(self):
        assert self.main_key, "Main metric is undefined. Please use `self.set_main()` first."
        return self.metrics[self.main_key]

    def set_main(self, key, downward=True):
        self.main_key = key
        self.downward = downward

    def __iter__(self):
        for key, value in self.metrics.items():
            yield key, value
