class Metrics:
    """The metrics class."""

    main_key: str
    downward: bool

    def __init__(self, *args, **kwargs):
        """
        Initialize Metrics object.

        Args:
            *args: Variable length arguments. Should be key-value pairs.
            **kwargs: Additional keyword arguments.
        """
        assert len(args) % 2 == 0, "Missing key or value, please make sure input is paired."
        self.data = {}
        self.num_samples = 0

        for kv in zip(args[0::2], args[1::2]):
            self.data[kv[0]] = kv[1]

        for k, v in kwargs.items():
            self.data[k] = v

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key] / self.num_samples

    def __len__(self):
        return len(self.data)

    def add(self, *args, **kwargs):
        """
        Add a value to a specific key.

        Args:
            *args: Additional key-value pairs to add.
            **kwargs: Additional keyword arguments.
        """
        assert len(args) % 2 == 0, "Missing key or value, please make sure input is paired."
        self.num_samples += 1
        for kv in zip(args[0::2], args[1::2]):
            self.data[kv[0]] += kv[1]
        for k, v in kwargs.items():
            self.data[k] += v

    def reset(self):
        """Reset all metrics and the number of samples."""
        for k in self.data:
            self.data[k] = 0
            self.num_samples = 0

    def set_main(self, key, downward=True):
        """
        Set the main key for comparison.

        Args:
            key: The main key.
            downward: Flag indicating whether lower values are better.
        """
        self.main_key = key
        self.downward = downward

    def __iter__(self):
        for key, value in self.data.items():
            yield key, self[key]

    @property
    def main_value(self):
        """Get the value of the main key."""
        assert self.main_key, "No main key, please use set_main() first."
        return self[self.main_key]

    def compare(self, value: float):
        """
        Compare the main value with another metric value.

        Args:
            value: The metric value to compare with.

        Returns:
            bool: True if this metric is better, False otherwise.
        """
        if self.downward and value > self.main_value:
            return True
        elif not self.downward and value < self.main_value:
            return True
        else:
            return False
