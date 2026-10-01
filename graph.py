# graph.py

from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from nodes import (
    compensation_agent_node,
    escalation_node,
    final_response_node,
    policy_checker_node,
    rebooking_agent_node,
    refund_agent_node,
    request_classifier_node,
    supervisor_approval_node,
)
from state import RebookingState
from tools import get_booking, search_flights


# 1. Routing Decision 1: Intent Router
def route_intent(state: RebookingState) -> Literal["rebooking_agent", "refund_agent", "compensation_agent", "escalation_node"]:
    intent = state.get("intent")
    if intent == "rebook":
        return "rebooking_agent"
    elif intent == "refund":
        return "refund_agent"
    elif intent == "compensation":
        return "compensation_agent"
    return "escalation_node"


# 2. Routing Decision 2: Rebooking Next Step (Tool vs Checker)
def route_rebooking_output(state: RebookingState) -> Literal["rebooking_tools", "policy_checker"]:
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "rebooking_tools"
    return "policy_checker"


# 3. Routing Decision 3: Policy Validation Router (Pass vs Retry vs Escalate)
def route_policy_result(
    state: RebookingState,
) -> Literal["final_response", "supervisor_approval", "rebooking_agent", "escalation_node"]:
    if state.get("policy_passed"):
        sol = state.get("proposed_solution") or {}
        # Route payouts over $300 to supervisor review
        if sol.get("type") in ["refund", "compensation"] and float(sol.get("amount", 0.0)) > 300.0:
            return "supervisor_approval"
        return "final_response"

    # Check retry limit
    if state.get("retry_count", 0) >= 3:
        return "escalation_node"

    # Send back to originating rebooking agent
    sol_type = (state.get("proposed_solution") or {}).get("type")
    if sol_type == "rebook":
        return "rebooking_agent"

    return "escalation_node"


# 4. Routing Decision 4: Supervisor Decision Router
def route_supervisor_decision(state: RebookingState) -> Literal["final_response", "escalation_node"]:
    if state.get("supervisor_approved"):
        return "final_response"
    return "escalation_node"


# --- Graph Construction ---
builder = StateGraph(RebookingState)

# Nodes
builder.add_node("classifier", request_classifier_node)
builder.add_node("rebooking_agent", rebooking_agent_node)
builder.add_node("rebooking_tools", ToolNode([get_booking, search_flights]))
builder.add_node("refund_agent", refund_agent_node)
builder.add_node("compensation_agent", compensation_agent_node)
builder.add_node("policy_checker", policy_checker_node)
builder.add_node("supervisor_approval", supervisor_approval_node)
builder.add_node("escalation_node", escalation_node)
builder.add_node("final_response", final_response_node)

# Flow Connections
builder.add_edge(START, "classifier")

# 1. Intent Router
builder.add_conditional_edges(
    "classifier",
    route_intent,
    {
        "rebooking_agent": "rebooking_agent",
        "refund_agent": "refund_agent",
        "compensation_agent": "compensation_agent",
        "escalation_node": "escalation_node",
    },
)

# 2. Rebooking tool loop
builder.add_conditional_edges(
    "rebooking_agent",
    route_rebooking_output,
    {
        "rebooking_tools": "rebooking_tools",
        "policy_checker": "policy_checker",
    },
)
builder.add_edge("rebooking_tools", "rebooking_agent")

# Solvers to Policy Checker
builder.add_edge("refund_agent", "policy_checker")
builder.add_edge("compensation_agent", "policy_checker")

# 3. Policy Checker conditional edge (INCLUDES 'supervisor_approval')
builder.add_conditional_edges(
    "policy_checker",
    route_policy_result,
    {
        "final_response": "final_response",
        "supervisor_approval": "supervisor_approval",
        "rebooking_agent": "rebooking_agent",
        "escalation_node": "escalation_node",
    },
)

# 4. Supervisor Approval conditional edge
builder.add_conditional_edges(
    "supervisor_approval",
    route_supervisor_decision,
    {
        "final_response": "final_response",
        "escalation_node": "escalation_node",
    },
)

builder.add_edge("escalation_node", "final_response")
builder.add_edge("final_response", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)