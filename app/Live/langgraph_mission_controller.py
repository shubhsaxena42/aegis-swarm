"""
LangGraph Drone Mission Controller (3-Agent Hierarchy)
══════════════════════════════════════════════════════

Implements the "Orchestrated Agentic Workflow" with a 3-agent hierarchy:
1. COORDINATOR (Supervisor):  Parses user intent and delegates tasks.
2. TACTICAL (Action):         Executes maneuvers (Follow, Patrol, Land).
3. ANALYST (Perception):      Monitors stats, assesses threats, identifies targets.

ARCHITECTURE:
┌──────────────┐      ┌─────────────┐      ┌──────────────┐
│     USER     │─────▶│ COORDINATOR │◀────▶│   ANALYST    │
└──────────────┘      └─────────────┘      └──────────────┘
                             ▲                    │
                             ▼                    ▼
                      ┌─────────────┐      ┌──────────────┐
                      │  TACTICAL   │      │    TOOLS     │
                      └─────────────┘      └──────────────┘

REQUIREMENTS:
    pip install langgraph langchain-openai langchain-core
"""

import os
import json
import asyncio
import operator
from typing import TypedDict, Annotated, Literal, List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ============================================================
# CONFIGURATION
# ============================================================

# DECISION_INTERVAL_SECONDS = 2.0
MODEL_NAME = "gpt-4o-mini"  # Fast, efficient model for real-time control

# ============================================================
# SHARED STATE (Between LangGraph and Low-Level Controller)
# ============================================================

class MissionMode(Enum):
    IDLE = "idle"
    FOLLOW = "follow"
    PATROL = "patrol"
    RETURN_HOME = "return_home"
    EMERGENCY = "emergency"

@dataclass
class DroneState:
    connected: bool = False
    flying: bool = False
    battery_percent: int = 100
    altitude_m: float = 0.0
    position: Dict[str, float] = None
    velocity_ms: float = 0.0
    mode: MissionMode = MissionMode.IDLE
    
    def __post_init__(self):
        if self.position is None: self.position = {"x": 0, "y": 0}
    
    def to_dict(self) -> Dict:
        return {
            "connected": self.connected, "flying": self.flying,
            "battery": self.battery_percent, "alt": self.altitude_m,
            "mode": self.mode.value
        }

@dataclass
class TargetState:
    track_id: int
    class_name: str
    confidence: float
    position_in_frame: Dict[str, float]
    size_in_frame: float
    is_locked: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)

class SharedState:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_state()
        return cls._instance
    
    def _init_state(self):
        self.drone = DroneState()
        self.targets: List[TargetState] = []
        self.locked_target_id: Optional[int] = None
        self.events: List[str] = []
        self.mode = MissionMode.IDLE
    
    def update_drone(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.drone, k): setattr(self.drone, k, v)

    def update_targets(self, targets: List[TargetState]):
        self.targets = targets

    def add_event(self, event: str):
        self.events.append(f"[{datetime.now().strftime('%H:%M:%S')}] {event}")
        if len(self.events) > 20: self.events = self.events[-20:]
    
    def get_context(self) -> str:
        t_summary = [f"#{t.track_id} {t.class_name}" + (" (LOCKED)" if t.is_locked else "") for t in self.targets]
        return f"""
STATUS: Mode={self.mode.value}, Bat={self.drone.battery_percent}%, Alt={self.drone.altitude_m}m
TARGETS: {', '.join(t_summary) if t_summary else 'None'}
"""

shared_state = SharedState()

# ============================================================
# TOOLS (Categorized by Agent)
# ============================================================

# --- Analyst Tools ---
@tool("get_drone_telemetry")
def get_drone_telemetry() -> str:
    """Get full telemetry including battery, altitude, velocity."""
    return json.dumps(shared_state.drone.to_dict())

@tool("scan_targets")
def scan_targets() -> str:
    """Scan and list all detected objects/targets in view."""
    return json.dumps([t.to_dict() for t in shared_state.targets])

@tool("analyze_threats")
def analyze_threats() -> str:
    """Analyze current situation for threats (low battery, obstacles)."""
    threats = []
    if shared_state.drone.battery_percent < 20: threats.append("CRITICAL BATTERY")
    if not shared_state.drone.connected: threats.append("DISCONNECTED")
    return f"THREATS: {threats}" if threats else "No immediate threats."

# --- Tactical Tools ---
@tool("engage_target")
def engage_target(track_id: int) -> str:
    """Lock onto and follow a specific target by ID."""
    targets = [t for t in shared_state.targets if t.track_id == track_id]
    if not targets: return f"Target #{track_id} not found."
    
    shared_state.locked_target_id = track_id
    shared_state.mode = MissionMode.FOLLOW
    return f"Engaging target #{track_id}."

@tool("execute_maneuver")
def execute_maneuver(maneuver: str) -> str:
    """Execute a named maneuver: 'return_home', 'land', 'patrol', 'hover'."""
    try:
        if maneuver == "land": shared_state.mode = MissionMode.EMERGENCY
        elif maneuver == "return_home": shared_state.mode = MissionMode.RETURN_HOME
        elif maneuver == "hover": shared_state.mode = MissionMode.IDLE
        elif maneuver == "patrol": shared_state.mode = MissionMode.PATROL
        else: return "Unknown maneuver."
        return f"Executing {maneuver}."
    except Exception as e: return str(e)

# ============================================================
# AGENT DEFINITIONS
# ============================================================

llm = ChatOpenAI(model=MODEL_NAME)

def create_agent(llm, tools, system_prompt):
    """Helper to create an agent node."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    if tools:
        llm_with_tools = llm.bind_tools(tools)
        return prompt | llm_with_tools
    return prompt | llm

# --- Analyst Agent ---
analyst_agent = create_agent(
    llm, 
    [get_drone_telemetry, scan_targets, analyze_threats], 
    "You are the ANALYST. Your job is perception and assessment. "
    "Use tools to gather data. Provide concise summaries to the Coordinator."
)

# --- Tactical Agent ---
tactical_agent = create_agent(
    llm, 
    [engage_target, execute_maneuver], 
    "You are the TACTICAL OFFICER. Your job is execution. "
    "Execute maneuvers and engage targets as directed. Confirm completion."
)

# --- Coordinator Agent ---
# Coordinator has no tools, just routing logic (implemented in edges)
coordinator_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the MISSION COORDINATOR. "
     "Delegate tasks: "
     "1. Ask ANALYST for status/targets. "
     "2. Command TACTICAL to execute actions. "
     "3. Synthesize final answer for user. "
     "Current Context: {context}"),
    MessagesPlaceholder(variable_name="messages"),
])
coordinator_chain = coordinator_prompt | llm

# ============================================================
# LANGGRAPH STATE & NODES
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    agent_scratchpad: List[BaseMessage]
    context: str

# Node: Coordinator
def coordinator_node(state: AgentState):
    ctx = shared_state.get_context()
    response = coordinator_chain.invoke({"messages": state["messages"], "context": ctx})
    return {"messages": [response], "context": ctx}

# Node: Analyst
def analyst_node(state: AgentState):
    # In a real agent loop, we'd handle tool calls here. 
    # For simplicity in this demo, we assume the LLM generates the tool call
    # and we execute it immediately if present, or return text.
    # This is a simplified "ReAct" step for the sub-agent.
    last_msg = state["messages"][-1]
    response = analyst_agent.invoke({
        "messages": state["messages"], 
        "agent_scratchpad": []
    })
    return {"messages": [response], "next_agent": "coordinator"}

# Node: Tactical
def tactical_node(state: AgentState):
    response = tactical_agent.invoke({
        "messages": state["messages"], 
        "agent_scratchpad": []
    })
    return {"messages": [response], "next_agent": "coordinator"}

# Node: Tool Executor (Generic)
def tool_node(state: AgentState):
    last_msg = state["messages"][-1]
    tool_map = {
        "get_drone_telemetry": get_drone_telemetry,
        "scan_targets": scan_targets,
        "analyze_threats": analyze_threats,
        "engage_target": engage_target,
        "execute_maneuver": execute_maneuver
    }
    
    results = []
    if hasattr(last_msg, "tool_calls"):
        for call in last_msg.tool_calls:
            tool_func = tool_map.get(call["name"])
            if tool_func:
                res = tool_func.invoke(call["args"])
                results.append(ToolMessage(content=str(res), tool_call_id=call["id"]))
    
    return {"messages": results}

# Logic: Router
def router(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    
    # If the last message has tool calls, go to tools
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    
    # If it's a ToolMessage, go back to the sender (simplified: Coordinator usually)
    if isinstance(last_msg, ToolMessage):
        return "coordinator" # In this simple flow, tools return info to flow
        
    # Coordinator routing logic based on text content
    if isinstance(last_msg, AIMessage):
        content = last_msg.content.lower()
        if "analyst" in content or "status" in content or "check" in content:
            return "analyst"
        if "tactical" in content or "engage" in content or "land" in content or "follow" in content:
            return "tactical"
        if "final answer" in content or "done" in content:
            return END
            
    return END

# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("coordinator", coordinator_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("tactical", tactical_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "coordinator")

# Conditional edges from Coordinator
def coordinator_router(state: AgentState):
    msg = state["messages"][-1]
    content = msg.content.lower()
    
    # Heuristic routing (Real impl would use function calling or structured output)
    if "analyst" in content or "check" in content or "scan" in content:
        return "analyst"
    elif "tactical" in content or "execute" in content or "engage" in content:
        return "tactical"
    return END

workflow.add_conditional_edges(
    "coordinator",
    coordinator_router,
    {"analyst": "analyst", "tactical": "tactical", END: END}
)

# Agents -> Tools -> Coordinator loop
workflow.add_conditional_edges(
    "analyst",
    lambda x: "tools" if x["messages"][-1].tool_calls else "coordinator",
    {"tools": "tools", "coordinator": "coordinator"}
)

workflow.add_conditional_edges(
    "tactical",
    lambda x: "tools" if x["messages"][-1].tool_calls else "coordinator",
    {"tools": "tools", "coordinator": "coordinator"}
)

workflow.add_edge("tools", "coordinator")

app = workflow.compile()

# ============================================================
# CONTROLLER CLASS
# ============================================================

class LangGraphMissionController:
    def __init__(self):
        self.app = app
        self.history = []

    def process_command(self, user_input: str) -> str:
        self.history.append(HumanMessage(content=user_input))
        
        inputs = {"messages": self.history, "next_agent": "", "context": "", "agent_scratchpad": []}
        
        # Run graph
        final_state = None
        for output in self.app.stream(inputs):
            for key, value in output.items():
                # print(f"  --> Node: {key}") # Debug
                final_state = value
        
        if final_state and final_state.get("messages"):
            response = final_state["messages"][-1]
            content = response.content
            self.history.append(AIMessage(content=content))
            return content
        return "No response."

# ============================================================
# MAIN / TEST
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY required for LangGraph Agents.")
    else:
        ctrl = LangGraphMissionController()
        print("🤖 System Ready. (Coordinator/Tactical/Analyst)")
        while True:
            try:
                cmd = input("Command: ")
                if cmd == "q": break
                print(f"🤖 {ctrl.process_command(cmd)}")
            except KeyboardInterrupt: break
