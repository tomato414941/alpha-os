from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParentOrder:
    symbol: str
    quantity: float
    slices: int


@dataclass(frozen=True)
class ChildOrder:
    symbol: str
    quantity: float


class EqualQuantityOrderSlicer:
    def slice(self, order: ParentOrder) -> tuple[ChildOrder, ...]:
        if order.slices <= 0:
            return ()

        child_quantity = order.quantity / order.slices
        return tuple(
            ChildOrder(symbol=order.symbol, quantity=child_quantity)
            for _ in range(order.slices)
        )


def slice_order(
    slicer: EqualQuantityOrderSlicer,
    order: ParentOrder,
) -> tuple[ChildOrder, ...]:
    return slicer.slice(order)


def main() -> None:
    child_orders = slice_order(
        EqualQuantityOrderSlicer(),
        ParentOrder(symbol="BTC", quantity=0.9, slices=3),
    )
    print(child_orders)


if __name__ == "__main__":
    main()
