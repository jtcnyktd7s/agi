import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# Simulation Logic
def simulate_feedback(memory_agents, reasoning_rate, meta_learning_rate, transfer_learning_rate, steps):
    """
    Simulate the feedback mechanisms for Memory, Reasoning, Meta-Learning, and Transfer-Learning agents.
    """
    # Initialize feedback arrays
    memory_feedback = np.zeros(steps)
    reasoning_feedback = np.zeros(steps)
    meta_learning_feedback = np.zeros(steps)
    transfer_learning_feedback = np.zeros(steps)

    # Set initial values
    memory_feedback[0] = memory_agents
    reasoning_feedback[0] = 20
    meta_learning_feedback[0] = -1.0
    transfer_learning_feedback[0] = 2

    # Simulate over time
    for t in range(1, steps):
        memory_feedback[t] = memory_feedback[t - 1] + np.random.choice([-1, 0, 1])
        reasoning_feedback[t] = reasoning_feedback[t - 1] + reasoning_rate
        meta_learning_feedback[t] = meta_learning_feedback[t - 1] - meta_learning_rate * np.random.rand()
        transfer_learning_feedback[t] = transfer_learning_feedback[t - 1] + transfer_learning_rate * np.random.rand()

    return memory_feedback, reasoning_feedback, meta_learning_feedback, transfer_learning_feedback


# Visualization
def plot_feedback_overlay(base_simulation, updated_simulation, steps):
    """
    Dynamically plot and overlay the updated simulation graphs on the base results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Overlay for Memory Agent Feedback
    axes[0, 0].plot(range(steps), base_simulation[0], color='blue', alpha=0.7, label="Base Memory")
    axes[0, 0].plot(range(steps), updated_simulation[0], color='orange', linestyle='dashed', label="Updated Memory")
    axes[0, 0].set_title("Memory Agent Feedback")
    axes[0, 0].set_xlabel("Time Steps")
    axes[0, 0].set_ylabel("Feedback")
    axes[0, 0].legend()

    # Overlay for Reasoning Agent Feedback
    axes[0, 1].plot(range(steps), base_simulation[1], color='green', alpha=0.7, label="Base Reasoning")
    axes[0, 1].plot(range(steps), updated_simulation[1], color='red', linestyle='dashed', label="Updated Reasoning")
    axes[0, 1].set_title("Reasoning Agent Feedback")
    axes[0, 1].set_xlabel("Time Steps")
    axes[0, 1].set_ylabel("Feedback")
    axes[0, 1].legend()

    # Overlay for Meta-Learning Feedback
    axes[1, 0].plot(range(steps), base_simulation[2], color='purple', alpha=0.7, label="Base Meta-Learning")
    axes[1, 0].plot(range(steps), updated_simulation[2], color='yellow', linestyle='dashed', label="Updated Meta-Learning")
    axes[1, 0].set_title("Meta-Learning Feedback")
    axes[1, 0].set_xlabel("Time Steps")
    axes[1, 0].set_ylabel("Feedback")
    axes[1, 0].legend()

    # Overlay for Transfer Learning Feedback
    axes[1, 1].plot(range(steps), base_simulation[3], color='purple', alpha=0.7, label="Base Transfer")
    axes[1, 1].plot(range(steps), updated_simulation[3], color='brown', linestyle='dashed', label="Updated Transfer")
    axes[1, 1].set_title("Transfer Learning Feedback")
    axes[1, 1].set_xlabel("Time Steps")
    axes[1, 1].set_ylabel("Feedback")
    axes[1, 1].legend()

    # Render the dynamic overlay plots
    st.pyplot(fig)


# Main App
def main():
    # Set Title
    st.title("AGI Feedback Simulator with Interactive Overlay")
    st.markdown("""
    This simulation models feedback mechanisms for:
    - Memory Agents,
    - Reasoning Agents,
    - Meta-Learning,
    - Transfer Learning.

    Use the sliders to adjust the **reasoning rate** and **meta-learning rate** parameters.
    The graphs will dynamically overlay the changes on the original simulation feedback trends.
    """)

    # Sidebar for base simulation parameters
    st.sidebar.header("Simulation Sliders")
    steps = st.sidebar.slider("Simulation Steps", 50, 200, 100)
    default_reasoning_rate = st.sidebar.slider("Reasoning Rate", 0.1, 2.0, 1.0)
    default_meta_learning_rate = st.sidebar.slider("Meta-Learning Rate", 0.01, 0.5, 0.1)

    # Run the base simulation first with default settings
    base_simulation = simulate_feedback(
        memory_agents=20,
        reasoning_rate=default_reasoning_rate,
        meta_learning_rate=default_meta_learning_rate,
        transfer_learning_rate=1.0,
        steps=steps
    )

    st.info("Use the sliders below to dynamically adjust simulation parameters.")

    # Sliders for user adjustments
    updated_reasoning_rate = st.slider("Adjust Reasoning Rate (dynamic)", 0.1, 3.0, default_reasoning_rate)
    updated_meta_learning_rate = st.slider("Adjust Meta-Learning Rate (dynamic)", 0.01, 0.5, default_meta_learning_rate)

    # Simulate updated feedback
    updated_simulation = simulate_feedback(
        memory_agents=20,
        reasoning_rate=updated_reasoning_rate,
        meta_learning_rate=updated_meta_learning_rate,
        transfer_learning_rate=1.0,
        steps=steps
    )

    # Render graph with dynamic overlay
    plot_feedback_overlay(base_simulation, updated_simulation, steps)


if __name__ == "__main__":
    main()
