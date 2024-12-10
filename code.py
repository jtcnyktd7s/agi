import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Simulation Logic
def run_simulation(memory_steps, reasoning_rate, meta_learning_rate, transfer_learning_rate):
    # Initialize arrays to track feedback over time
    memory_agent_feedback = np.zeros(memory_steps)
    reasoning_agent_feedback = np.zeros(memory_steps)
    meta_learning_feedback = np.zeros(memory_steps)
    transfer_learning_feedback = np.zeros(memory_steps)

    # Initialize starting conditions
    memory_agent_feedback[0] = 7  # Start memory agents with count 7
    reasoning_agent_feedback[0] = 20  # Starting feedback count for reasoning agents
    meta_learning_feedback[0] = -1  # Start meta-learning feedback at -1
    transfer_learning_feedback[0] = 2  # Start transfer learning knowledge count at 2

    # Run simulation steps
    for t in range(1, memory_steps):
        # Memory agent feedback with small stochastic updates
        memory_agent_feedback[t] = memory_agent_feedback[t - 1] + np.random.choice([0, 1])

        # Reasoning agent feedback grows linearly (deterministic growth influenced by reasoning_rate)
        reasoning_agent_feedback[t] = reasoning_agent_feedback[t - 1] + reasoning_rate

        # Meta-learning feedback with stochastic degradation over time
        meta_learning_feedback[t] = meta_learning_feedback[t - 1] - meta_learning_rate * np.random.rand()

        # Transfer learning feedback increases but with slight variability
        transfer_learning_feedback[t] = transfer_learning_feedback[t - 1] + transfer_learning_rate * np.random.choice([0, 1])

    # Return all feedback arrays
    return memory_agent_feedback, reasoning_agent_feedback, meta_learning_feedback, transfer_learning_feedback


# Plotting logic
def plot_results(memory_feedback, reasoning_feedback, meta_feedback, transfer_feedback):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # Plot Memory Agent Feedback
    ax[0, 0].plot(memory_feedback, label="Memory Agent Feedback")
    ax[0, 0].set_title("Memory Agent Feedback")
    ax[0, 0].set_xlabel("Time Steps")
    ax[0, 0].set_ylabel("Agent Count")
    ax[0, 0].legend()

    # Plot Reasoning Agent Feedback
    ax[0, 1].plot(reasoning_feedback, label="Reasoning Agent Feedback", color="orange")
    ax[0, 1].set_title("Reasoning Agent Feedback")
    ax[0, 1].set_xlabel("Time Steps")
    ax[0, 1].set_ylabel("Agent Feedback")
    ax[0, 1].legend()

    # Plot Meta-Learning Feedback
    ax[1, 0].plot(meta_feedback, label="Meta-Learning Feedback", color="red")
    ax[1, 0].set_title("Meta-Learning Feedback")
    ax[1, 0].set_xlabel("Time Steps")
    ax[1, 0].set_ylabel("Feedback")
    ax[1, 0].legend()

    # Plot Transfer Learning Feedback
    ax[1, 1].plot(transfer_feedback, label="Transfer Learning Feedback", color="green")
    ax[1, 1].set_title("Transfer Learning Feedback")
    ax[1, 1].set_xlabel("Time Steps")
    ax[1, 1].set_ylabel("Feedback")
    ax[1, 1].legend()

    st.pyplot(fig)


# Main Streamlit App
st.title("AGI Feedback Simulation Dashboard")
st.write(
    "Explore how memory agents, reasoning agents, meta-learning, and transfer learning feedback patterns interact over time."
)

# Input sliders for simulation parameters
st.sidebar.header("Simulation Parameters")
memory_steps = st.sidebar.slider("Number of Time Steps", min_value=10, max_value=100, value=50)
reasoning_rate = st.sidebar.slider("Reasoning Agent Feedback Rate", min_value=1, max_value=10, value=2)
meta_learning_rate = st.sidebar.slider("Meta-Learning Rate", min_value=0.01, max_value=0.2, value=0.05)
transfer_learning_rate = st.sidebar.slider("Transfer Learning Rate", min_value=0.01, max_value=0.2, value=0.05)

# Run simulation button
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        # Run the simulation with provided parameters
        memory_feedback, reasoning_feedback, meta_feedback, transfer_feedback = run_simulation(
            memory_steps, reasoning_rate, meta_learning_rate, transfer_learning_rate
        )

        # Display the summary stats
        correlation_memory_reasoning = np.corrcoef(memory_feedback, reasoning_feedback)[0, 1]
        correlation_meta_transfer = np.corrcoef(meta_feedback, transfer_feedback)[0, 1]

        # Display statistics
        st.write("Simulation Statistics Summary:")
        st.write(f"Memory-Agent vs Reasoning-Agent Correlation: {correlation_memory_reasoning:.2f}")
        st.write(f"Meta-Learning vs Transfer Learning Correlation: {correlation_meta_transfer:.2f}")

        # Plot Results
        plot_results(memory_feedback, reasoning_feedback, meta_feedback, transfer_feedback)
else:
    st.warning("Adjust parameters and click 'Run Simulation' to see results.")
