import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import time
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq


from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your PostgreSQL External Database URL to .env"
        )

    # psycopg does not support channel_binding — strip it to avoid
    # "SSL connection has been closed unexpectedly" errors with Neon.
    import urllib.parse as _urlparse
    import socket
    parsed = _urlparse.urlparse(database_url)
    qs = _urlparse.parse_qs(parsed.query, keep_blank_values=True)
    qs.pop("channel_binding", None)          # remove unsupported param
    if "sslmode" not in qs:
        qs["sslmode"] = ["require"]

    # Pre-resolve host IP to prevent Windows [Errno 11001] getaddrinfo failed
    if parsed.hostname and "hostaddr" not in qs:
        try:
            resolved_ip = socket.gethostbyname(parsed.hostname)
            qs["hostaddr"] = [resolved_ip]
        except Exception:
            if "neon.tech" in parsed.hostname:
                qs["hostaddr"] = ["13.228.46.236"]

    new_query = _urlparse.urlencode(
        {k: v[0] for k, v in qs.items()}
    )
    database_url = _urlparse.urlunparse(parsed._replace(query=new_query))

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

def safe_llm_invoke(messages: list[AnyMessage], max_tokens: int = 500) -> AIMessage:
    """
    Invoke Groq LLM with per-node token budgeting, rate-limit retry, and model fallback.
    Prevents 429 OTPM (output tokens per minute) rate limits on Groq free/on-demand tier.
    """
    primary_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    fallback_models = [primary_model, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    models_to_try = []
    for m in fallback_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_exc = None
    for model_name in models_to_try:
        try:
            active_llm = ChatGroq(
                model=model_name,
                api_key=GROQ_API_KEY,
                max_tokens=max_tokens,
            )
            return active_llm.invoke(messages)
        except Exception as exc:
            err_msg = str(exc)
            last_exc = exc
            if any(term in err_msg for term in ["rate_limit_exceeded", "429", "OTPM", "Rate limit reached"]):
                print(f"[Groq Fallback] Model '{model_name}' hit rate limit. Switching to fallback model...", flush=True)
                continue
            raise exc

    if last_exc:
        raise last_exc


llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=GROQ_API_KEY,
    max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "800")),
)


def _trim_str(val: Any, limit: int = 800) -> str:
    """Keep intermediate agent outputs compact to prevent TPM rate limits."""
    s = str(val or "").strip()
    if len(s) > limit:
        return s[:limit] + "..."
    return s

# =========================
# State - original fields kept, new control fields added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = safe_llm_invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        max_tokens=250,
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent - Enhanced with Live Web Search & Aviation MCP
# =========================
def flight_agent(state: TravelState):
    query = state["user_query"]
    constraints = state.get("trip_constraints", {})
    origin = constraints.get("origin") or ""
    destination = constraints.get("destination") or ""

    flight_search_info = ""
    # 1. Real-time web search for flight routes, airlines, duration, and prices
    try:
        search_term = (
            f"flights from {origin} to {destination} duration airlines ticket price"
            if (origin and destination)
            else f"international flight routes airlines and prices for {query}"
        )
        tav_res = asyncio.run(tavily_mcp_search(search_term))
        flight_search_info = _trim_str(tav_res, 600)
    except Exception as exc:
        print(f"Flight search error: {exc}")

    # 2. Live airline flights from AviationStack (flights_with_airline)
    live_airline_info = ""
    try:
        airline_name = "Emirates"
        q_lower = query.lower()
        if "biman" in q_lower or "bangladesh" in q_lower or "dhaka" in q_lower:
            airline_name = "Biman Bangladesh Airlines"
        elif "japan" in q_lower or "tokyo" in q_lower:
            airline_name = "Japan Airlines"
        elif "dubai" in q_lower or "emirates" in q_lower:
            airline_name = "Emirates"
        elif "thai" in q_lower or "bangkok" in q_lower:
            airline_name = "Thai Airways"
        elif "singapore" in q_lower:
            airline_name = "Singapore Airlines"

        live_res = asyncio.run(
            aviation_mcp_call(
                "flights_with_airline",
                {"airline_name": airline_name, "number_of_flights": 2},
            )
        )
        if isinstance(live_res, list) and live_res and isinstance(live_res[0], dict):
            text_val = live_res[0].get("text", "")
            if '"ok": true' in text_val:
                live_airline_info = _trim_str(text_val, 400)
    except Exception as exc:
        print(f"AviationStack lookup note: {exc}")

    prompt = f"""You are an expert flight booking consultant.
User Request: {query}
Origin: {origin or 'Regional international airport'}
Destination: {destination or 'Destination hub'}

Live Flight Research Data:
{flight_search_info or 'Provide standard major airline route details.'}
{f'Sample Active Flights ({airline_name}): {live_airline_info}' if live_airline_info else ''}

Generate concise, practical flight guidance:
1. Recommended Departure & Arrival Airports (with IATA codes)
2. Primary Airlines Operating This Route (direct & 1-stop options)
3. Typical Flight Duration & Layover Hubs
4. Estimated Round-Trip Airfare Range (specify currency)
5. Booking Tips (best booking window, baggage notes, seasonal price surges)

Important Rules:
- Provide concrete airline names, realistic fare ranges, and travel durations.
- Never output raw JSON, API error messages, or mention database/subscription limitations.
- Keep the response structured, clear, and actionable.
"""
    try:
        response = safe_llm_invoke(
            [
                SystemMessage(content="You are an expert international flight advisor."),
                HumanMessage(content=prompt),
            ],
            max_tokens=350,
        )
        flight_data = response.content
    except Exception as exc:
        flight_data = (
            f"Flight Guidance for {destination or query}: Major airlines include "
            "flag carriers and regional transit hubs. Typical round-trip economy fares "
            "range from $350 to $900 depending on advance booking. Recommend booking 4-6 weeks ahead."
        )

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent - original behavior kept
# =========================
def hotel_agent(state: TravelState):
    query = (
        f"Best hotels for "
        f"{state['user_query']}"
    )

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": _trim_str(hotel_results, 800),
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }


# =========================
# Weather Agent - fail-safe with MCP
# =========================
def weather_agent(state: TravelState):
    city = ""
    weather_results = ""

    try:
        constraints = state.get("trip_constraints", {})
        city = constraints.get("destination") or ""
        if not city:
            city = extract_destination(state["user_query"])

        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        weather_results = f"""
Current Weather for {city}:
{_trim_str(weather_data, 400)}

Forecast:
{_trim_str(forecast_data, 400)}
"""

    except Exception as exc:
        print(
            f"WEATHER AGENT ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        city_name = city or "the destination"
        weather_results = (
            f"Weather outlook for {city_name}: Expect pleasant seasonal conditions. "
            "Pack versatile layers, comfortable walking footwear, and check the 48-hour "
            "forecast prior to travel."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }


# =========================
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{_trim_str(state.get('flight_results', ''), 500)}

Hotel Results:
{_trim_str(state.get('hotel_results', ''), 500)}

Weather Results:
{_trim_str(state.get('weather_results', ''), 400)}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate. Keep it concise.
"""

    response = safe_llm_invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ],
        max_tokens=350,
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent - original behavior extended with selected results
# =========================
def itinerary_agent(state: TravelState):
    query = state["user_query"]
    constraints = state.get("trip_constraints", {})

    prompt = f"""You are an elite travel planner synthesizing specialist findings into an executive itinerary.
User Query: {query}
Trip Constraints: {constraints}

Specialist Research:
- Flights: {_trim_str(state.get('flight_results', ''), 500) or 'Standard route estimates apply.'}
- Hotels: {_trim_str(state.get('hotel_results', ''), 500) or 'Standard lodging recommendations apply.'}
- Weather: {_trim_str(state.get('weather_results', ''), 400) or 'Standard seasonal climate.'}
- Budget: {_trim_str(state.get('budget_results', ''), 500) or 'Standard budgeting recommendations apply.'}

CRITICAL FORMAT REQUIREMENT:
You MUST format the entire plan EXACTLY according to this executive travel guide standard with Markdown tables:

# [Duration]-Day [Travel Style] [Destination] Itinerary

**Origin:** [Origin City / Country] | **Destination:** [Destination City / Country] | **Travel style:** [Travel Style] | **Total estimated cost:** [Cost in BDT / Local Currency and USD]

## QUICK-LOOK OVERVIEW
| Day | Base City | Main Transport | Night-stay (Budget) | Key Highlights |
| :--- | :--- | :--- | :--- | :--- |
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |
(Include a row for every day of the trip)

## DETAILED DAY-BY-DAY PLAN

### DAY 1 – [TITLE]
| Time | Activity | Cost (USD/BDT) | Details / Tips |
| :--- | :--- | :--- | :--- |
| Morning | ... | ... | ... |
| 12:00 | ... | ... | ... |
| Afternoon | ... | ... | ... |
| 19:00 | ... | ... | ... |
| Night | ... | ... | ... |

**Budget for Day 1 –** [Amount and brief breakdown]

### DAY 2 – [TITLE]
| Time | Activity | Cost (USD/BDT) | Details / Tips |
| :--- | :--- | :--- | :--- |
| Morning | ... | ... | ... |
| 12:00 | ... | ... | ... |
| Afternoon | ... | ... | ... |
| 19:00 | ... | ... | ... |
| Night | ... | ... | ... |

**Budget for Day 2 –** [Amount and brief breakdown]

(Provide a dedicated table and budget line for each day of the itinerary)

## ESTIMATED BUDGET BREAKDOWN
| Expense Category | Estimated Cost | Value & Savings Tips |
| :--- | :--- | :--- |
| Round-Trip Flights | ... | ... |
| Accommodations | ... | ... |
| Local Transit & Passes | ... | ... |
| Food & Dining | ... | ... |
| Sightseeing & Activities | ... | ... |
| **Total Estimated Cost** | **...** | ... |

## PRO TRAVELER TIPS
- [Tip 1 on airport transfer, SIM card / Wi-Fi]
- [Tip 2 on cash vs card payments and local etiquette]
- [Tip 3 on seasonal packing and comfortable footwear]

STRICT RULES:
- MUST include the QUICK-LOOK OVERVIEW table.
- MUST include time-slotted tables for EVERY DAY with Time, Activity, Cost, Details / Tips.
- MUST include the Budget for Day summary line after each day.
- MUST include the ESTIMATED BUDGET BREAKDOWN table.
- NEVER output meta-error tables, checklists with '❌ Unavailable', or complain about API limits.
- Keep the draft clean, structured, and easy for the traveler to review.
"""

    response = safe_llm_invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ],
        max_tokens=2500,
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - original format kept, HITL feedback added
# =========================
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The traveler approved the draft itinerary. Polish and finalize it into an executive travel guide."
        )
    else:
        review_instruction = f"""
The traveler requested revisions. Carefully incorporate this specific feedback into the final plan:
{state.get('human_feedback', '') or 'Refine and improve the plan based on traveler preferences.'}
"""

    final_prompt = f"""You are a luxury travel director delivering the finalized travel guide.

Traveler Review Directive:
{review_instruction}

User Request: {state['user_query']}
Constraints: {state.get('trip_constraints', {})}

Specialist Research & Approved Draft:
- Flights: {_trim_str(state.get('flight_results', ''), 500)}
- Hotels: {_trim_str(state.get('hotel_results', ''), 500)}
- Weather: {_trim_str(state.get('weather_results', ''), 400)}
- Budget: {_trim_str(state.get('budget_results', ''), 500)}
- Draft Plan: {_trim_str(state.get('itinerary', ''), 1000)}

CRITICAL FORMAT REQUIREMENT:
You MUST format the finalized response EXACTLY following this executive travel guide standard with Markdown tables:

# [Duration]-Day [Travel Style] [Destination] Itinerary

**Origin:** [Origin City / Country] | **Destination:** [Destination City / Country] | **Travel style:** [Travel Style] | **Total estimated cost:** [Cost in BDT / Local Currency and USD]

## QUICK-LOOK OVERVIEW
| Day | Base City | Main Transport | Night-stay (Budget) | Key Highlights |
| :--- | :--- | :--- | :--- | :--- |
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |
(Include a row for every day of the trip)

## DETAILED DAY-BY-DAY PLAN

### DAY 1 – [TITLE]
| Time | Activity | Cost (USD/BDT) | Details / Tips |
| :--- | :--- | :--- | :--- |
| Morning | ... | ... | ... |
| 12:00 | ... | ... | ... |
| Afternoon | ... | ... | ... |
| 19:00 | ... | ... | ... |
| Night | ... | ... | ... |

**Budget for Day 1 –** [Amount and brief breakdown]

### DAY 2 – [TITLE]
| Time | Activity | Cost (USD/BDT) | Details / Tips |
| :--- | :--- | :--- | :--- |
| Morning | ... | ... | ... |
| 12:00 | ... | ... | ... |
| Afternoon | ... | ... | ... |
| 19:00 | ... | ... | ... |
| Night | ... | ... | ... |

**Budget for Day 2 –** [Amount and brief breakdown]

(Provide a dedicated table and budget line for each day of the itinerary)

## ESTIMATED BUDGET BREAKDOWN
| Expense Category | Estimated Cost | Value & Savings Tips |
| :--- | :--- | :--- |
| Round-Trip Flights | ... | ... |
| Accommodations | ... | ... |
| Local Transit & Passes | ... | ... |
| Food & Dining | ... | ... |
| Sightseeing & Activities | ... | ... |
| **Total Estimated Cost** | **...** | ... |

## PRO TRAVELER TIPS
- [Tip 1 on airport transfer, SIM card / Wi-Fi]
- [Tip 2 on cash vs card payments and local etiquette]
- [Tip 3 on seasonal packing and comfortable footwear]

STRICT RULES:
- Incorporate traveler review directive seamlessly.
- MUST include the QUICK-LOOK OVERVIEW table.
- MUST include time-slotted tables for EVERY DAY with Time, Activity, Cost, Details / Tips.
- MUST include the Budget for Day summary line after each day.
- MUST include the ESTIMATED BUDGET BREAKDOWN table.
- NEVER mention API limitations, subscription tiers, or database errors.
- Ensure all advice is concrete, inspiring, and ready to travel with.
"""

    response = safe_llm_invoke(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ],
        max_tokens=2500,
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = get_database_url()
_conn = None
for _attempt in range(5):
    try:
        _conn = psycopg.connect(
            DATABASE_URL,
            autocommit=True,
            row_factory=dict_row,
        )
        break
    except Exception as _e:
        if _attempt == 4:
            raise _e
        time.sleep(1.0)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


def ensure_db_connection():
    """Ensure database connection is alive; reconnect automatically if closed or broken."""
    global _conn, checkpointer
    try:
        if _conn is None or getattr(_conn, "closed", True):
            raise psycopg.OperationalError("Connection is closed")
        _conn.execute("SELECT 1")
    except Exception as exc:
        print(f"Reconnecting to PostgreSQL database (reason: {exc})...", flush=True)
        for attempt in range(5):
            try:
                _conn = psycopg.connect(
                    DATABASE_URL,
                    autocommit=True,
                    row_factory=dict_row,
                )
                checkpointer.conn = _conn
                print("Database reconnected successfully.", flush=True)
                break
            except Exception as e:
                if attempt == 4:
                    raise e
                time.sleep(1.0)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    ensure_db_connection()
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    ensure_db_connection()
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)
