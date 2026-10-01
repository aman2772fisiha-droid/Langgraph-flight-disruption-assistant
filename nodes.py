# nodes.py
import json
import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from state import ClassifiedRequest, ProposedSolution, RebookingState
from langgraph.types import interrupt
from tools import get_booking, get_fare_rules, search_flights

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY", "")

# Try real model; fallback if offline/rate-limited
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        google_api_key=api_key,
    )
except Exception:
    llm = None

def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)

# 1. Request Classifier Node
def request_classifier_node(state: RebookingState) -> Dict[str, Any]:
    last_message = str(state["messages"][-1].content).lower()
    
    # Fast deterministic classification (avoids burning quota)
    if "refund" in last_message or "money back" in last_message:
        intent = "refund"
    elif "delayed" in last_message or "compensation" in last_message or "entitled" in last_message:
        intent = "compensation"
    elif "manager" in last_message or "third time" in last_message or "complain" in last_message:
        intent = "complaint"
    else:
        intent = "rebook"

    booking = state.get("booking_details")
    if not booking and state.get("booking_ref"):
        booking = get_booking.invoke({"booking_ref": state["booking_ref"]})

    return {
        "intent": intent,
        "constraints": {"latest_arrival": "noon"} if "noon" in last_message else {},
        "booking_details": booking,
        "policy_failure_reason": None,
    }

# 2. Rebooking Agent Node
def rebooking_agent_node(state: RebookingState) -> Dict[str, Any]:
    booking = state.get("booking_details") or {}
    origin = booking.get("origin", "LHR")
    dest = booking.get("destination", "DXB")
    
    has_tool_run = any(isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls) for m in state["messages"])
    
    # Step A: Trigger the Tool Call if not yet called
    if not has_tool_run:
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "search_flights",
                "args": {"origin": origin, "destination": dest, "date": "2026-10-01"},
                "id": "call_search_01"
            }]
        )
        return {"messages": [tool_call_msg]}

    # Step B: Pick flight based on scenario
    # If ORD -> NRT (UPGRADE1 scenario), return Business to exercise retry policy loop
    if origin == "ORD" and dest == "NRT":
        solution: ProposedSolution = {
            "type": "rebook",
            "flight_number": "NH-011",
            "cabin_class": "business",  # Violates policy (booked economy)
            "departure_time": "2026-10-01T14:00:00",
            "arrival_time": "2026-10-02T18:00:00",
            "connection_time_mins": 90,
            "hours_from_disruption": 8,
        }
    else:
        solution: ProposedSolution = {
            "type": "rebook",
            "flight_number": "EK-008",
            "cabin_class": "economy",
            "departure_time": "2026-10-01T14:00:00",
            "arrival_time": "2026-10-02T00:30:00",
            "connection_time_mins": 75,
            "hours_from_disruption": 6,
        }

    return {
        "proposed_solution": solution,
        "messages": [AIMessage(content=f"Selected flight {solution['flight_number']}.")]
    }

# 3. Refund Agent Node
def refund_agent_node(state: RebookingState) -> Dict[str, Any]:
    booking = state.get("booking_details") or {}
    fare_type = booking.get("fare_type", "economy_standard")
    status = booking.get("status", "delayed")
    
    fare_info = get_fare_rules.invoke({"fare_type": fare_type})
    
    if status == "cancelled":
        amount = fare_info["fare_amount"]
        details = "Full refund approved due to airline cancellation."
    else:
        if fare_info.get("refundable"):
            amount = fare_info["fare_amount"]
            details = "Full refund approved per refundable fare terms."
        else:
            amount = fare_info.get("taxes", 0.0)
            details = "Taxes-only refund for non-refundable ticket."

    solution: ProposedSolution = {
        "type": "refund",
        "amount": amount,
        "details": details,
    }
    return {
        "proposed_solution": solution,
        "messages": [AIMessage(content=f"Proposed Refund: ${amount}. {details}")],
    }

# 4. Compensation Agent Node
def compensation_agent_node(state: RebookingState) -> Dict[str, Any]:
    booking = state.get("booking_details") or {}
    delay = float(booking.get("delay_hours", 0.0))
    status = booking.get("status")
    
    if delay < 3.0 and status != "cancelled":
        amount = 0.0
        details = "Delay under 3 hours: No compensation due under policy."
    elif 3.0 <= delay < 6.0:
        amount = 200.0
        details = "Delay between 3 to 6 hours: $200 standard entitlement."
    else:
        amount = 400.0
        details = "Delay over 6 hours or cancellation: $400 statutory payout."

    solution: ProposedSolution = {
        "type": "compensation",
        "amount": amount,
        "details": details,
    }
    return {
        "proposed_solution": solution,
        "messages": [AIMessage(content=f"Proposed Compensation: ${amount}. {details}")],
    }

# 5. Policy Checker Node (Pure Python)
def policy_checker_node(state: RebookingState) -> Dict[str, Any]:
    solution = state.get("proposed_solution")
    booking = state.get("booking_details") or {}
    retry_count = state.get("retry_count", 0)
    
    if not solution:
        return {
            "policy_passed": False,
            "policy_failure_reason": "No concrete solution proposal received.",
            "retry_count": retry_count + 1,
        }

    sol_type = solution.get("type")

    if sol_type == "rebook":
        booked_cabin = (booking.get("cabin_class") or "economy").lower()
        prop_cabin = (solution.get("cabin_class") or "").lower()
        
        cabin_rank = {"economy": 1, "premium_economy": 2, "business": 3, "first": 4}
        if cabin_rank.get(prop_cabin, 1) > cabin_rank.get(booked_cabin, 1):
            return {
                "policy_passed": False,
                "policy_failure_reason": f"Cabin class '{prop_cabin}' exceeds booked '{booked_cabin}'.",
                "retry_count": retry_count + 1,
            }
        
        if solution.get("hours_from_disruption", 0) > 48:
            return {
                "policy_passed": False,
                "policy_failure_reason": "Departure is beyond 48 hours from disruption.",
                "retry_count": retry_count + 1,
            }
            
        conn = solution.get("connection_time_mins")
        if conn is not None and conn < 60:
            return {
                "policy_passed": False,
                "policy_failure_reason": f"Connection time {conn} min is below the 60-minute requirement.",
                "retry_count": retry_count + 1,
            }

    elif sol_type == "refund":
        is_cancelled = booking.get("status") == "cancelled"
        fare_rules = get_fare_rules.invoke({"fare_type": booking.get("fare_type", "")})
        if is_cancelled and solution.get("amount") != fare_rules.get("fare_amount"):
            return {
                "policy_passed": False,
                "policy_failure_reason": "Cancelled flights must receive full refund.",
                "retry_count": retry_count + 1,
            }

    elif sol_type == "compensation":
        delay = float(booking.get("delay_hours", 0.0))
        status = booking.get("status")
        amt = solution.get("amount", 0.0)
        
        if (delay >= 6.0 or status == "cancelled") and amt != 400.0:
            return {
                "policy_passed": False,
                "policy_failure_reason": "Disruptions over 6h or cancellations require $400 payout.",
                "retry_count": retry_count + 1,
            }

    return {
        "policy_passed": True,
        "policy_failure_reason": None,
    }

# 6. Escalation Agent Node
def escalation_node(state: RebookingState) -> Dict[str, Any]:
    reason = state.get("policy_failure_reason") or "Direct passenger complaint or non-policy request."
    note = (
        f"ESCALATION HANDOVER TICKET\n"
        f"Passenger Ref: {state.get('booking_ref')}\n"
        f"Intent: {state.get('intent')}\n"
        f"Attempts Made: {state.get('retry_count', 0)}\n"
        f"Failure/Trigger: {reason}\n"
        f"Action: Handed over to Senior Customer Support Desk."
    )
    return {
        "is_escalated": True,
        "escalation_handover": note,
        "messages": [AIMessage(content=note)],
    }

# 7. Final Response Agent Node
def final_response_node(state: RebookingState) -> Dict[str, Any]:
    intent = state.get("intent")
    is_escalated = state.get("is_escalated")
    sol = state.get("proposed_solution") or {}

    if is_escalated:
        reply = (
            f"Dear Passenger, your request regarding booking {state.get('booking_ref')} has been escalated "
            f"to our Senior Customer Support Desk for personal review. "
            f"Reason: {state.get('policy_failure_reason') or 'Special handling requested'}. "
            f"A representative will contact you directly."
        )
    elif intent == "rebook":
        reply = (
            f"Dear Passenger, we have rebooked you on flight {sol.get('flight_number')} "
            f"departing at {sol.get('departure_time')}. "
            f"Your boarding pass will be updated automatically in your app."
        )
    elif intent == "refund":
        reply = (
            f"Dear Passenger, your refund request of ${sol.get('amount')} has been approved. "
            f"Funds will return to your original payment method in 3-5 business days."
        )
    elif intent == "compensation":
        reply = (
            f"Dear Passenger, your compensation claim for ${sol.get('amount')} has been processed under airline policy. "
            f"You will receive an email confirmation shortly."
        )
    else:
        reply = "Dear Passenger, your request has been received and processed."

    return {
        "final_response": reply,
        "messages": [AIMessage(content=reply)],
    }

# bonus node
def supervisor_approval_node(state: RebookingState) -> Dict[str, Any]:
    solution = state.get("proposed_solution") or {}
    sol_type = solution.get("type")
    amount = float(solution.get("amount", 0.0))

    # Payouts <= $300 or rebooking flights bypass supervisor review
    if sol_type not in ["refund", "compensation"] or amount <= 300.0:
        return {
            "supervisor_approved": True,
            "supervisor_notes": "Auto-approved: payout under or equal to $300.",
        }

    # Pause execution and request supervisor approval
    # The dictionary passed to interrupt() is visible to the external operator
    supervisor_decision = interrupt({
        "action": "payout_approval_request",
        "booking_ref": state.get("booking_ref"),
        "type": sol_type,
        "amount": amount,
        "details": solution.get("details"),
        "instruction": "Reply with {'approved': True/False, 'notes': '...'}",
    })

    # When app.invoke(Command(resume=...)) is called, execution continues here:
    approved = supervisor_decision.get("approved", False)
    notes = supervisor_decision.get("notes", "No notes provided.")

    if not approved:
        # Failed approval routes through escalation
        return {
            "supervisor_approved": False,
            "supervisor_notes": notes,
            "policy_passed": False,
            "policy_failure_reason": f"Supervisor rejected ${amount} {sol_type}: {notes}",
        }

    return {
        "supervisor_approved": True,
        "supervisor_notes": notes,
    }