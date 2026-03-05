class StrategyResult:
    def __init__(self, signal_type, trigger, entry, stop, target, r_multiple):
        self.signal_type = signal_type
        self.trigger = trigger
        self.entry = entry
        self.stop = stop
        self.target = target
        self.r_multiple = r_multiple


class BaseStrategy:
    name = "base"

    def generate(self, candle_data: dict) -> StrategyResult | None:
        raise NotImplementedError
