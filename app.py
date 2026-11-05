"""Website for the AI-Based Adaptive Traffic Signal Management Agent (Streamlit).
It uses the same engine as the Tkinter app (traffic_agent.py), so behaviour and results are identical.
Run locally:  streamlit run app.py
"""
import random
import streamlit as st
import traffic_agent as ta
import webview as wv

st.set_page_config(page_title="Adaptive Traffic Signal Agent", page_icon="🚦", layout="wide")
st.markdown("<style>.block-container{padding-top:1.2rem}</style>", unsafe_allow_html=True)

@st.cache_resource
def get_agents():
    return ta.load_agents()

@st.cache_data(show_spinner=False)
def cached_eval(seeds):
    saved = {k: v for k, v in ta.SCENARIOS.items() if k == "Custom"}
    ta.SCENARIOS.pop("Custom", None)                    # the evaluation always uses the four standard scenarios
    try:
        return ta.run_evaluation(seeds)
    finally:
        ta.SCENARIOS.update(saved)

agents = get_agents()

st.title("🚦 AI-Based Adaptive Traffic Signal Management Agent")
st.caption("Fixed-time vs Search + Rules vs DQN, on a simulated four-way intersection. Vehicles arrive at random; nothing is entered by hand.")

# ------------------------------ button callbacks ------------------------------
# Callbacks run BEFORE the page is redrawn, so the button label always matches the real state.
st.session_state.setdefault("running", False)
def toggle_run():
    st.session_state.running = not st.session_state.running
def request_reset():
    st.session_state.running = False      # set before the label is drawn, so it reads "Start"
    st.session_state.need_reset = True

# ------------------------------ sidebar controls ------------------------------
with st.sidebar:
    st.header("Simulation")
    controller = st.selectbox("Controller", ta.CONTROLLERS, index=1, key="ctl", on_change=request_reset)
    scenario = st.selectbox("Traffic scenario", list(k for k in ta.SCENARIOS if k != "Custom") + ["Custom"], index=2, key="scn", on_change=request_reset)
    rates = {}
    if scenario == "Custom":
        st.caption("Arrival probability per second for each approach (arrivals are still random).")
        for a in "NSEW":
            rates[a] = st.slider(f"{a} approach", 0.0, 0.5, 0.15, 0.01, on_change=request_reset)
        ta.SCENARIOS["Custom"] = lambda t, r=dict(rates): dict(r)
    speed = st.slider("Speed (simulated seconds per refresh)", 1, 30, 5)
    c1, c2 = st.columns(2)
    c1.button("⏸ Pause" if st.session_state.running else "▶ Start", on_click=toggle_run, use_container_width=True)
    c2.button("↺ Reset", on_click=request_reset, use_container_width=True)
    run_eval = st.button("📊 Run Evaluation (5 runs)", use_container_width=True)
    if not agents:
        st.warning("DQN weights not found. Only Fixed-time and Search + Rules will work. Add dqn_alone.npz and dqn_shield.npz next to app.py.")

# ------------------------------ session state ------------------------------
key = (controller, scenario, tuple(sorted(rates.items())))
if "sim" not in st.session_state or st.session_state.get("key") != key or st.session_state.get("need_reset"):
    st.session_state.sim = ta.make_sim(scenario, controller, random.randint(0, 9999), agents)
    st.session_state.key = key
    st.session_state.running = False
    st.session_state.need_reset = False
sim = st.session_state.sim

# ------------------------------ live view ------------------------------
@st.fragment(run_every=0.5)
def live(speed):
    sim = st.session_state.sim
    if st.session_state.running:
        for _ in range(speed): sim.step()
    left, right = st.columns([1, 1.15])
    with left:
        st.markdown(wv.intersection_svg(sim), unsafe_allow_html=True)
        st.caption("Green / red / yellow (all-red) signals at each corner; blue blocks are queued vehicles (up to 9 drawn per approach).")
    with right:
        st.code("\n".join(wv.status_lines(sim, controller, bool(agents))), language=None)
        rows = wv.candidate_rows(sim)
        if rows:
            st.markdown("**Candidate phases** (score = queue cost for Search, Q-value for DQN)")
            st.table(rows)
        else:
            st.markdown("**Candidate phases:** not used by the fixed-time controller.")
        md = wv.metrics_dict(sim)
        cols = st.columns(3)
        for i, (k, v) in enumerate(md.items()):
            cols[i % 3].metric(k, v)
    st.progress(min(sim.t / ta.RUN_SECONDS, 1.0), text=f"Simulated time: {sim.t}s of {ta.RUN_SECONDS}s "
                + ("(running)" if st.session_state.running else "(paused)"))

live(speed)

# ------------------------------ evaluation ------------------------------
if run_eval:
    with st.spinner("Running all controllers on all scenarios (5 runs of 1,800 s each)..."):
        res = cached_eval(5)
    st.subheader("Evaluation results (mean of 5 runs)")
    st.dataframe(wv.eval_table(res), use_container_width=True, hide_index=True)
    st.caption("The report uses 30 runs (python traffic_agent.py --eval). Rule violations = decisions that broke a safety rule.")

with st.expander("About the controllers"):
    st.markdown("""
- **Fixed-time:** swaps the green phase every 30 s whatever the traffic.
- **Search + Rules:** scores *keep* and *switch* as queue cost + switching penalty (5) and picks the lowest valid score.
- **DQN alone:** a small neural network (4-32-32-2, NumPy) trained offline picks the action with the highest Q-value; rules are not enforced, broken rules are counted.
- **DQN + Shield:** the same idea trained with a rule shield, so actions that break the minimum-green (10 s), maximum-green (40 s) or conflict rules are never chosen.
""")
