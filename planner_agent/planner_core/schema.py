# planner_core/schema.py
from __future__ import annotations
from pydantic import BaseModel, Field, validator, field_validator
from typing import Dict, List, Optional


class Geo(BaseModel):
    latitude: float
    longitude: float


class TicketPrice(BaseModel):
    adult: Optional[float] = 0.0


class Candidate(BaseModel):
    place_id: str
    name: str
    type: str
    tags: Optional[List[str]] = []
    geo: Geo
    opening_hours: Optional[Dict[str, object]] = {}
    rating: Optional[float] = 0.0
    low_carbon_score: Optional[float] = 0.0
    ticket_price_sgd: Optional[TicketPrice] = TicketPrice()
    geo_cluster_id: Optional[str] = None


class TripDates(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class OptionalReq(BaseModel):
    interests: Optional[List[str]] = []
    uninterests: Optional[List[str]] = []
    diet: Optional[str] = None
    slot_times: Optional[Dict[str,str]] = None
    type_caps: Optional[Dict[str,int]] = None
    attractions_per_day: Optional[int] = 2
    accommodation_location: Optional[Dict[str, float]] = None # {lat, lon}

class Requirements(BaseModel):
    duration_days: Optional[int] = None
    trip_dates: Optional[TripDates] = TripDates()
    budget_total_sgd: Optional[float] = None
    weights: Optional[Dict[str, float]] = None
    optional: OptionalReq = OptionalReq()

    @validator('duration_days', pre=True, always=True)

    def _nonzero_days(cls, v):
        if v is None:
            return 0
        return int(v)


class Retrieval(BaseModel):
    candidates: List[Candidate]


class PlannerPayload(BaseModel):
    requirements: Requirements
    retrieval: Retrieval