from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# Structured output schemas
class ClassifiedRequest(BaseModel):
    intent: Literal["rebook", "refund", "compensation", "complaint"] = Field(
        description="The primary intent of the passenger."
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted constraints such as latest_arrival_time, preferred_cabin, etc.",
    )

class ProposedSolution(TypedDict, total=False):
    type: Literal["rebook", "refund", "compensation"]
    flight_number: Optional[str]
    cabin_class: Optional[str]
    departure_time: Optional[str]
    arrival_time: Optional[str]
    connection_time_mins: Optional[int]
    hours_from_disruption: Optional[int]
    amount: Optional[float]
    details: Optional[str]

class RebookingState(TypedDict):
    # Message history with reducer
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Request tracking
    request_id: str
    booking_ref: str
    intent: Optional[Literal["rebook", "refund", "compensation", "complaint"]]
    constraints: Dict[str, Any]
    
    # Domain entities
    booking_details: Optional[Dict[str, Any]]
    proposed_solution: Optional[ProposedSolution]
    
    # Verification & loops
    policy_passed: Optional[bool]
    policy_failure_reason: Optional[str]
    retry_count: int
    is_escalated: bool
    escalation_handover: Optional[str]

    # ... existing fields ...
    supervisor_approved: Optional[bool]
    supervisor_notes: Optional[str]
    
    # Final output
    final_response: Optional[str]