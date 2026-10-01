# run_scenarios.py
import os
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from graph import app


def run_test(
    scenario_title: str,
    thread_id: str,
    booking_ref: str,
    message: str,
    is_followup: bool = False,
):
    print(f"\n{'='*70}\n{scenario_title}\n{'='*70}")
    config = {"configurable": {"thread_id": thread_id}}

    input_payload = {
        "messages": [HumanMessage(content=message)],
    }
    if not is_followup:
        input_payload.update({
            "request_id": f"REQ-{thread_id}",
            "booking_ref": booking_ref,
            "retry_count": 0,
            "is_escalated": False,
        })

    path_traversed = []

    # 1. Stream initial execution until completion or an interrupt() pause
    for event in app.stream(input_payload, config=config, stream_mode="updates"):
        for node_name in event.keys():
            path_traversed.append(node_name)

    # 2. Check if execution paused on a supervisor approval interrupt
    state = app.get_state(config)
    if state.next and any(task.interrupts for task in state.tasks):
        for task in state.tasks:
            for interrupt_item in task.interrupts:
                info = interrupt_item.value
                print("\n[HUMAN-IN-THE-LOOP SUPERVISOR REVIEW REQUIRED]")
                print(f"Booking Ref     : {info.get('booking_ref')}")
                print(f"Payout Type     : {info.get('type')}")
                print(f"Payout Amount   : ${info.get('amount')}")
                print(f"Context Details : {info.get('details')}")

        # 3. Interactive prompt in the terminal
        choice = ""
        while choice not in ["y", "n", "yes", "no"]:
            choice = input("\nSupervisor Decision - Approve this payout? (y/n): ").strip().lower()

        if choice in ["y", "yes"]:
            decision = {
                "approved": True,
                "notes": "Verified against flight logs. Payout authorized by Supervisor.",
            }
            print("-> Decision recorded: [APPROVED]")
        else:
            decision = {
                "approved": False,
                "notes": "Denied: Delay caused by bad weather at the destination airport (extraordinary circumstances).",
            }
            print("-> Decision recorded: [REJECTED: Bad Weather at Destination]")

        # 4. Resume graph execution with the supervisor's decision
        for event in app.stream(Command(resume=decision), config=config, stream_mode="updates"):
            for node_name in event.keys():
                path_traversed.append(node_name)

    # 5. Output summary
    final_snapshot = app.get_state(config)
    print("\n--- Execution Summary ---")
    print(f"Path Traversed      : {' -> '.join(path_traversed)}")
    print(f"Intent Decided      : {final_snapshot.values.get('intent')}")
    print(f"Policy Passed       : {final_snapshot.values.get('policy_passed')}")
    print(f"Supervisor Approved : {final_snapshot.values.get('supervisor_approved')}")
    print(f"Escalated           : {final_snapshot.values.get('is_escalated')}")
    print(f"\nFinal Response to Passenger:\n{final_snapshot.values.get('final_response')}")


if __name__ == "__main__":
    # Scenario 1: Rebooking (No payout > $300, runs automatically)
    run_test(
        "SCENARIO 1: Cancelled Flight - Must Arrive By Noon",
        thread_id="thread_s1",
        booking_ref="XK9L2P",
        message="My flight to Dubai was cancelled. I need to be there by tomorrow noon.",
    )

    # Scenario 1 Follow-Up: Refund on Same Thread ($750 refund - WILL PROMPT YOU)
    run_test(
        "SCENARIO 1 (FOLLOW-UP): Same Thread - Change to Refund ($750)",
        thread_id="thread_s1",
        booking_ref="",
        message="Actually, can I get a refund instead?",
        is_followup=True,
    )

    # Scenario 2: Compensation ($400 compensation - WILL PROMPT YOU)
    run_test(
        "SCENARIO 2: 6-Hour Delay - Compensation Inquiry ($400)",
        thread_id="thread_s2",
        booking_ref="DL600X",
        message="My flight is delayed 6 hours. Am I entitled to anything?",
    )

    # Scenario 3: Policy Violation Loop (Runs automatically)
    run_test(
        "SCENARIO 3: Policy Violation Loop - Only Business Class Available",
        thread_id="thread_s3",
        booking_ref="UPGRADE1",
        message="My flight was cancelled. Rebook me on the next available flight.",
    )

    # Scenario 4: Direct Complaint (Runs automatically)
    run_test(
        "SCENARIO 4: Escalation - Unhappy Customer Requesting Manager",
        thread_id="thread_s4",
        booking_ref="COMPLAIN9",
        message="This is the third time you've cancelled on me. I want to speak to a manager.",
    )