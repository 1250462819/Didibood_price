"""SQLAlchemy models for price analytics."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Integer, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from db.session import Base


class PriceMarketSnapshot(Base):
    __tablename__ = "price_market_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "city",
            "purpose",
            "property_type",
            "neighborhood",
            "period_start",
            name="uq_price_market_snapshot_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    purpose: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    property_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    neighborhood: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    sample_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    median_price_irr: Mapped[Decimal | None] = mapped_column(Numeric(20, 0), nullable=True)
    min_price_irr: Mapped[Decimal | None] = mapped_column(Numeric(20, 0), nullable=True)
    max_price_irr: Mapped[Decimal | None] = mapped_column(Numeric(20, 0), nullable=True)
    median_price_per_sqm_irr: Mapped[Decimal | None] = mapped_column(Numeric(20, 0), nullable=True)

    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="manual")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
