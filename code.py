import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# Simulation Logic
def simulate_system(memory_agents, reasoning_rate, meta_learning_rate, transfer_learning_rate, steps):
    """
    Simulate the system based on provided parameters.
    Args:
        memory_agents: Number of memory agents (feedback system for simulation).
        reasoning_rate: Reasoning agent feedback rate.
        meta_learning_rate: Degradation rate for meta-learning.
        transfer_learning_rate: Feedback rate change for transfer learning agents.
        steps: Number of simulation steps to run.
    Returns:
        memory_feedback: Memory agent feedback simulation data.
        reasoning_feedback: Reasoning agent feedback simulation data.
        meta_learning_feedback: Meta-learning feedback degradation data.
        transfer_learning_feedback: Transfer learning feedback data over time.
    """
    memory_feedback = []
    reasoning_feedback = []
    meta_learning_feedback = []
    transfer_learning_feedback = []

    # Initialize values
    mem_agents = memory_agents
    reason_feedback = 0
    meta_feedback = -1.0
    trans_feedback = 0

    for step in range(steps):
        # Simulate memory feedback with random variability
        mem_agents += np.random.randint(0, 2)  # Simulate stochastic feedback loop
        memory_feedback.append(mem_agents)

        # Simulate reasoning feedback (steady deterministic growth)
        reason_feedback += reasoning_rate
        reasoning_feedback.append(reason_feedback)

        # Simulate meta-learning degradation
        meta_feedback -= meta_learning_rate
        meta_learning_feedback.append(meta_feedback)

        # Simulate transfer learning feedback growth
        trans_feedback += transfer_learning_rate
        transfer_learning_feedback.append(trans_feedback)

    return memory_feedback, reasoning_feedback, meta_learning_feedback, transfer_learning_feedback


# Streamlit UI
st.title("AGI Simulation Feedback Model")
st.sidebar.header("Adjust Simulation Parameters")

# Sidebar sliders for simulation parameters
memory_agents_input = st.sidebar.slider("Number of Initial Memory Agents", min_value=5, max_value=30, value=10)
reasoning_rate_input = st.sidebar.slider("Reasoning Agent Feedback Rate", min_value=5, max_value=20, value=10)
meta_learning_rate_input = st.sidebar.slider("Meta-Learning Degradation Rate", min_value=0.01, max_value=0.2, value=0.1)
transfer_learning_rate_input = st.sidebar.slider("Transfer Learning Feedback Rate", min_value=1, max_value=5, value=2)
steps_input = st.sidebar.slider("Number of Simulation Steps", min_value=10, max_value=50, value=30)

# Run simulation logic
st.write("Running simulation with given parameters...")
memory_feedback, reasoning_feedback, meta_learning_feedback, transfer_learning_feedback = simulate_system(
    memory_agents_input,
    reasoning_rate_input,
    meta_learning_rate_input,
    transfer_learning_rate_input,
    steps_input,
)

# Display simulation feedback
st.subheader("Simulation Results Summary")
st.write("Memory Agent Feedback:", memory_feedback)
st.write("Reasoning Agent Feedback:", reasoning_feedback)
st.write("Meta-Learning Feedback:", meta_learning_feedback)
st.write("Transfer Learning Feedback:", transfer_learning_feedback)

# Graphs to visualize results
st.subheader("Simulation Graphs")
col1, col2 = st.columns(2)

# Graph 1: Memory Agent Feedback
with col1:
    st.write("Memory Agent Feedback")
    plt.plot(range(len(memory_feedback)), memory_feedback, label="Memory Agents")
    plt.xlabel("Time Steps")
    plt.ylabel("Agent Feedback")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

# Graph 2: Reasoning Agent Feedback
with col2:
    st.write("Reasoning Agent Feedback")
    plt.plot(range(len(reasoning_feedback)), reasoning_feedback, label="Reasoning Feedback", color='orange')
    plt.xlabel("Time Steps")
    plt.ylabel("Feedback Level")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

# Graph 3: Meta-Learning Feedback
st.write("Meta-Learning Feedback (Degradation)")
plt.plot(range(len(meta_learning_feedback)), meta_learning_feedback, label="Meta-Learning Feedback", color='red')
plt.xlabel("Time Steps")
plt.ylabel("Meta-Learning Feedback")
plt.legend()
st.pyplot(plt)
plt.clf()

# Graph 4: Transfer Learning Feedback
st.write("Transfer Learning Feedback")
plt.plot(range(len(transfer_learning_feedback)), transfer_learning_feedback, label="Transfer Learning Feedback", color='green')
plt.xlabel("Time Steps")
plt.ylabel("Feedback Level")
plt.legend()
st.pyplot(plt)
plt.clf()
