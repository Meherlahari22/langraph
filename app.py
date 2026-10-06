
import os
import re
import subprocess
import tempfile
from typing import TypedDict, Optional, List

import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. LLM INITIALIZATION
# ============================================================

def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Set it as an environment variable or in Streamlit secrets."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=api_key,
        temperature=0.2,
    )


llm = get_llm()


# ============================================================
# 2. STATE DEFINITION
# ============================================================

class VerilogState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    specification: Optional[str]
    verilog_code: Optional[str]
    testbench_code: Optional[str]
    simulation_result: Optional[str]
    report: Optional[str]
    iteration: int


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def clean_code(text: str, language: str = "verilog") -> str:
    """Remove Markdown code fences and return only source code."""
    if not text:
        return ""

    text = str(text).strip()

    # Remove ```verilog, ```systemverilog, ```sv, ```text, etc.
    text = re.sub(r"^```(?:verilog|systemverilog|sv|v|text)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def extract_response_text(response) -> str:
    """Safely extract text from LangChain/Gemini response."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


# ============================================================
# 4. TOOLS
# ============================================================

@tool
def run_verilog_simulation(verilog_code: str, testbench_code: str) -> str:
    """
    Compile and simulate Verilog RTL and its testbench using Icarus Verilog.
    Returns compiler errors, runtime output, or successful simulation output.
    """

    verilog_code = clean_code(verilog_code)
    testbench_code = clean_code(testbench_code)

    with tempfile.TemporaryDirectory() as temp_dir:
        design_file = os.path.join(temp_dir, "design.v")
        tb_file = os.path.join(temp_dir, "tb.v")
        output_file = os.path.join(temp_dir, "simulation.out")

        with open(design_file, "w", encoding="utf-8") as f:
            f.write(verilog_code)

        with open(tb_file, "w", encoding="utf-8") as f:
            f.write(testbench_code)

        # Check whether Icarus Verilog is installed.
        try:
            subprocess.run(
                ["iverilog", "-V"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            return (
                "IVERILOG_NOT_FOUND\n"
                "Icarus Verilog is not installed or is not available in PATH.\n"
                "Install it before running simulations."
            )
        except Exception as e:
            return f"IVERILOG_CHECK_ERROR\n{e}"

        # Compile.
        compile_process = subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-o",
                output_file,
                design_file,
                tb_file,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

        compile_output = (
            (compile_process.stdout or "") +
            (compile_process.stderr or "")
        ).strip()

        if compile_process.returncode != 0:
            return (
                "COMPILATION_FAILED\n\n"
                + (compile_output if compile_output else "Unknown compilation error.")
            )

        # Simulate.
        simulation_process = subprocess.run(
            ["vvp", output_file],
            capture_output=True,
            text=True,
            timeout=20,
        )

        simulation_output = (
            (simulation_process.stdout or "") +
            (simulation_process.stderr or "")
        ).strip()

        if simulation_process.returncode != 0:
            return (
                "SIMULATION_FAILED\n\n"
                + (simulation_output if simulation_output else "Unknown simulation error.")
            )

        return (
            "SIMULATION_PASSED\n\n"
            + (simulation_output if simulation_output else "Simulation completed successfully.")
        )


@tool
def generate_verilog_testbench(specification: str, verilog_code: str) -> str:
    """Generate a self-checking Verilog testbench for the generated RTL."""

    prompt = f"""
You are an expert Verilog verification engineer.

USER SPECIFICATION:
{specification}

VERILOG RTL:
{verilog_code}

Create a self-checking Verilog/SystemVerilog-compatible testbench.

Requirements:
1. Instantiate the correct DUT/module.
2. Generate all required clocks and resets.
3. Apply normal test cases.
4. Apply important edge cases.
5. Use assertions or explicit if/else checks.
6. Print PASS/FAIL messages.
7. Finish using $finish.
8. Do not use unavailable vendor-specific libraries.
9. Return ONLY testbench source code.
10. Do not use Markdown code fences.
"""

    response = llm.invoke(prompt)
    return clean_code(extract_response_text(response))


# ============================================================
# 5. GRAPH NODES
# ============================================================

def specification_node(state: VerilogState):
    specification = state["messages"][-1].content

    return {
        "specification": specification,
        "iteration": 0,
        "next_step": "developer",
    }


def verilog_developer_node(state: VerilogState):
    """Generate or repair Verilog RTL."""

    specification = state["specification"]
    previous_code = state.get("verilog_code")
    simulation_result = state.get("simulation_result")
    iteration = state.get("iteration", 0)

    if previous_code and simulation_result:
        prompt = f"""
You are an expert RTL design engineer debugging Verilog.

USER REQUIREMENT:
{specification}

PREVIOUS VERILOG CODE:
{previous_code}

SIMULATION / COMPILATION RESULT:
{simulation_result}

Fix the RTL so that it correctly satisfies the requirement and passes simulation.

Important:
- Preserve the intended module interface unless the previous interface is clearly wrong.
- Fix syntax errors.
- Fix width/sign/logic errors.
- Fix clock/reset behavior.
- Fix functional errors indicated by the simulation.
- Return ONLY valid Verilog/SystemVerilog source code.
- Do NOT use Markdown fences.
"""
    else:
        prompt = f"""
You are an expert Verilog RTL design engineer.

Design synthesizable Verilog/SystemVerilog for this requirement:

{specification}

Requirements:
- Create a clean synthesizable RTL design.
- Use a clear module name.
- Use appropriate input/output widths.
- Handle reset and clock correctly when required.
- Avoid vendor-specific primitives.
- Return ONLY the Verilog/SystemVerilog source code.
- Do NOT include explanations.
- Do NOT use Markdown code fences.
"""

    response = llm.invoke(prompt)
    code = clean_code(extract_response_text(response))

    return {
        "verilog_code": code,
        "iteration": iteration + 1,
        "next_step": "tester",
    }


def testbench_node(state: VerilogState):
    """Generate a verification testbench."""

    specification = state["specification"]
    verilog_code = state["verilog_code"]

    testbench = generate_verilog_testbench.invoke({
        "specification": specification,
        "verilog_code": verilog_code,
    })

    return {
        "testbench_code": clean_code(testbench),
        "next_step": "simulator",
    }


def simulator_node(state: VerilogState):
    """Compile and simulate the generated design."""

    result = run_verilog_simulation.invoke({
        "verilog_code": state["verilog_code"],
        "testbench_code": state["testbench_code"],
    })

    return {
        "simulation_result": result,
        "next_step": "manager",
    }


def manager_node(state: VerilogState):
    """
    Decide whether the design passed, needs another repair iteration,
    or should finish.
    """

    result = state.get("simulation_result", "")
    iteration = state.get("iteration", 0)

    if result.startswith("SIMULATION_PASSED"):
        decision = "success"
    elif iteration < 3:
        decision = "retry"
    else:
        decision = "failed"

    report = f"""
VERILOG AGENT REPORT
====================

Iteration: {iteration}
Status: {decision.upper()}

Simulation Result:
{result}
""".strip()

    return {
        "report": report,
        "next_step": decision,
    }


def retry_node(state: VerilogState):
    """Prepare state for another developer/debugging cycle."""
    return {
        "next_step": "developer"
    }


def success_node(state: VerilogState):
    return {
        "next_step": "end"
    }


def failed_node(state: VerilogState):
    return {
        "next_step": "end"
    }


# ============================================================
# 6. ROUTING
# ============================================================

def route_after_manager(state: VerilogState):
    decision = state.get("next_step")

    if decision == "retry":
        return "retry"

    if decision == "success":
        return "success"

    return "failed"


# ============================================================
# 7. GRAPH CONSTRUCTION
# ============================================================

workflow = StateGraph(VerilogState)

workflow.add_node("specification", specification_node)
workflow.add_node("developer", verilog_developer_node)
workflow.add_node("testbench", testbench_node)
workflow.add_node("simulator", simulator_node)
workflow.add_node("manager", manager_node)
workflow.add_node("retry", retry_node)
workflow.add_node("success", success_node)
workflow.add_node("failed", failed_node)

workflow.add_edge(START, "specification")
workflow.add_edge("specification", "developer")
workflow.add_edge("developer", "testbench")
workflow.add_edge("testbench", "simulator")
workflow.add_edge("simulator", "manager")

workflow.add_conditional_edges(
    "manager",
    route_after_manager,
    {
        "retry": "retry",
        "success": "success",
        "failed": "failed",
    },
)

workflow.add_edge("retry", "developer")
workflow.add_edge("success", END)
workflow.add_edge("failed", END)

verilog_agent = workflow.compile()


# ============================================================
# 8. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Verilog LangGraph Agent",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Verilog RTL LangGraph Agent")
st.write(
    "Generate Verilog RTL, automatically create a testbench, "
    "compile/simulate with Icarus Verilog, and repair errors."
)

with st.sidebar:
    st.header("Agent Settings")
    max_iterations = st.slider(
        "Maximum repair iterations",
        min_value=1,
        max_value=5,
        value=3,
    )

    st.info(
        "The agent uses Gemini for RTL generation and verification. "
        "Icarus Verilog is required for local simulation."
    )

default_task = """Design a 4-bit synchronous up counter.

Requirements:
- Module name: counter_4bit
- Inputs: clk and reset
- reset is synchronous and active high
- Output: 4-bit count
- Count increments on every rising edge of clk
- When reset is high, count becomes 0
"""

task = st.text_area(
    "Enter your Verilog design requirement",
    value=default_task,
    height=220,
)

run_agent = st.button(
    "🚀 Run Verilog Agent",
    type="primary",
    use_container_width=True,
)

if run_agent:
    if not task.strip():
        st.error("Please enter a Verilog requirement.")
        st.stop()

    try:
        # Temporarily override the manager retry limit through a wrapper
        # by running the graph normally. The built-in manager allows 3 tries.
        # For most projects, this is sufficient.
        initial_state: VerilogState = {
            "messages": [HumanMessage(content=task)],
            "next_step": None,
            "specification": None,
            "verilog_code": None,
            "testbench_code": None,
            "simulation_result": None,
            "report": None,
            "iteration": 0,
        }

        with st.spinner("Running Verilog agent..."):
            result = verilog_agent.invoke(
                initial_state,
                config={"recursion_limit": max_iterations * 6 + 10},
            )

        st.success("Agent execution completed.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧩 Generated Verilog RTL")
            st.code(
                result.get("verilog_code", ""),
                language="verilog",
            )

        with col2:
            st.subheader("🧪 Generated Testbench")
            st.code(
                result.get("testbench_code", ""),
                language="verilog",
            )

        st.subheader("🔬 Simulation Result")

        simulation_result = result.get("simulation_result", "")

        if simulation_result.startswith("SIMULATION_PASSED"):
            st.success(simulation_result)
        elif "IVERILOG_NOT_FOUND" in simulation_result:
            st.warning(simulation_result)
        else:
            st.error(simulation_result)

        st.subheader("📋 Agent Report")
        st.text(result.get("report", "No report generated."))

        st.download_button(
            "Download Verilog RTL",
            data=result.get("verilog_code", ""),
            file_name="design.v",
            mime="text/plain",
        )

        st.download_button(
            "Download Testbench",
            data=result.get("testbench_code", ""),
            file_name="tb.v",
            mime="text/plain",
        )

    except Exception as e:
        st.error(f"Agent error: {e}")
        st.exception(e)
