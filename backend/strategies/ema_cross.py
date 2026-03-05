from backend.strategies.base import BaseStrategy, StrategyResult

ATR_STOP_MULT   = 1.5
ATR_TARGET_MULT = 3.0
R_MULTIPLE      = round(ATR_TARGET_MULT / ATR_STOP_MULT, 2)  # 2.0


class EMACrossStrategy(BaseStrategy):
    name = "ema_cross"

    def generate(self, candle_data: dict) -> StrategyResult | None:
        ema_20     = candle_data["ema_20"]
        ema_50     = candle_data["ema_50"]
        prev_ema20 = candle_data.get("prev_ema_20")
        prev_ema50 = candle_data.get("prev_ema_50")
        atr_14     = candle_data.get("atr_14") or 0
        close      = candle_data["close"]

        golden = (
            prev_ema20 is not None and prev_ema50 is not None
            and prev_ema20 <= prev_ema50 and ema_20 > ema_50
        )
        death = (
            prev_ema20 is not None and prev_ema50 is not None
            and prev_ema20 >= prev_ema50 and ema_20 < ema_50
        )

        if golden:
            return StrategyResult(
                signal_type="long",
                trigger="golden_cross",
                entry=close,
                stop=round(close - ATR_STOP_MULT * atr_14, 4),
                target=round(close + ATR_TARGET_MULT * atr_14, 4),
                r_multiple=R_MULTIPLE,
            )
        if death:
            return StrategyResult(
                signal_type="short",
                trigger="death_cross",
                entry=close,
                stop=round(close + ATR_STOP_MULT * atr_14, 4),
                target=round(close - ATR_TARGET_MULT * atr_14, 4),
                r_multiple=R_MULTIPLE,
            )
        return None
