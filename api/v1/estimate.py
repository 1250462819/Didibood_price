"""Price estimate routes."""
from __future__ import annotations

from application.estimate.estimate import estimate_price
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import get_db
from schemas.price import PriceEstimateRequest, PriceEstimateResponse

router = APIRouter(prefix="/estimate", tags=["Estimate"])


@router.post("", response_model=PriceEstimateResponse)
def price_estimate(
    payload: PriceEstimateRequest,
    db: Session = Depends(get_db),
) -> PriceEstimateResponse:
    """Estimate property price from location and physical attributes."""
    return estimate_price(db, payload)
