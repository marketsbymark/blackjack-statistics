import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import simulation logic from the existing script
from bankroll_sim import validate_pmf, SimParams, simulate

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Blackjack Bankroll Simulator",
    layout="wide",
    page_icon="🎰",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# UI Header
# -----------------------------------------------------------------------------
st.title("🎰 Blackjack Bankroll Monte Carlo Simulator")
st.markdown("""
Welcome to the interactive Blackjack Bankroll Simulator! Adjust the parameters in the sidebar to run a 
Monte Carlo simulation estimating how long your bankroll will last under realistic casino rules.

**Rules Modeled:** 6-deck shoe, Dealer hits on soft 17 (H17), Double after split allowed (DAS), 
Blackjack pays 3:2, perfect basic strategy, flat betting.
""")

# -----------------------------------------------------------------------------
# Sidebar: User Inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    
    bet = st.number_input("Dollars per hand", min_value=1.0, value=25.0, step=5.0)
    bankroll = st.number_input("Starting bankroll ($)", min_value=1.0, value=500.0, step=50.0)
    minutes_per_hand = st.number_input("Minutes per hand", min_value=0.1, value=0.75, step=0.1)
    
    st.divider()
    
    sims = st.selectbox(
        "Number of simulations", 
        options=[1000, 5000, 10000, 20000, 50000], 
        index=3,
        help="Higher numbers provide smoother curves but take longer to compute."
    )
    max_hands = st.number_input("Max hands per sim", min_value=1000, value=20000, step=1000)
    seed = st.number_input("Random seed", value=20260424)
    n_trajectories = st.slider(
        "Trajectories to plot", 
        min_value=10, max_value=1000, value=100, step=10, 
        help="How many sample paths to draw. High values may lag the browser."
    )

    st.divider()
    
    run_sim = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Run Simulation
# -----------------------------------------------------------------------------
if run_sim or "result" not in st.session_state:
    with st.spinner("Rolling the dice (running simulation)..."):
        # Validate PMF and get expected values
        ev, sd = validate_pmf()
        
        # Build parameters
        params = SimParams(
            bet=float(bet),
            bankroll=float(bankroll),
            minutes_per_hand=float(minutes_per_hand),
            sims=int(sims),
            max_hands=int(max_hands),
            seed=int(seed),
            n_trajectories=int(n_trajectories)
        )
        
        # Run core logic
        result = simulate(params, ev, sd)
        
        # Cache results in session state so changing tabs/interacting doesn't re-run
        st.session_state["result"] = result
        st.session_state["params"] = params
        st.session_state["ev"] = ev
        st.session_state["sd"] = sd

# -----------------------------------------------------------------------------
# Extract Cache for Rendering
# -----------------------------------------------------------------------------
result = st.session_state["result"]
params = st.session_state["params"]
ev = st.session_state["ev"]
sd = st.session_state["sd"]

# Calculate top-level stats
ruin_rate = float(result.ruined.mean())
dollar_loss_per_hand = -ev * params.bet
hands_per_hour = 60.0 / params.minutes_per_hand
ev_dollar_loss_per_hour = dollar_loss_per_hand * hands_per_hour
survivors = result.ending_bankroll[~result.ruined]
mean_survivor_bankroll = float(np.mean(survivors)) if survivors.size else 0.0

# -----------------------------------------------------------------------------
# Dashboard Top Row: Key Metrics
# -----------------------------------------------------------------------------
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Overall Ruin Probability", 
    value=f"{ruin_rate * 100:.1f}%", 
    help="Chance of losing the ability to make the minimum bet before reaching max hands."
)
col2.metric(
    label="House Edge", 
    value=f"{-ev * 100:.2f}%",
    delta=f"EV: {ev:.4f} units/hand",
    delta_color="inverse"
)
col3.metric(
    label="Expected Loss per Hour", 
    value=f"${ev_dollar_loss_per_hour:.2f}",
    help=f"At {hands_per_hour:.1f} hands/hour and ${params.bet}/hand."
)
col4.metric(
    label="Mean Survivor Bankroll", 
    value=f"${mean_survivor_bankroll:,.2f}",
    help="Average bankroll for the players who did not hit ruin."
)

st.divider()

# -----------------------------------------------------------------------------
# Dashboard Row 2: Charts (Trajectories & Histogram)
# -----------------------------------------------------------------------------
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    n_samples = result.sample_paths.shape[0]
    
    st.subheader(f"📈 Sample Bankroll Trajectories ({n_samples} plotted)")
    st.markdown("Visualizing a random subset of player bankrolls over time.")
    
    fig_traj = go.Figure()
    
    x_axis = np.arange(result.sample_paths.shape[1])
    
    # Add each sample path
    for i in range(n_samples):
        y_vals = result.sample_paths[i]
        fig_traj.add_trace(go.Scatter(
            x=x_axis,
            y=y_vals,
            mode='lines',
            line=dict(width=1, color='rgba(0, 204, 150, 0.4)'), # Plotly Teal/Green semi-transparent
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Ruin threshold line
    fig_traj.add_hline(
        y=params.bet, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Ruin Line",
        annotation_position="bottom right"
    )
    
    # 50th Percentile (Median)
    p50_path = np.median(result.sample_paths, axis=0)
    fig_traj.add_trace(go.Scatter(
        x=x_axis, y=p50_path, mode='lines', 
        line=dict(color='white', width=3), 
        name='50th Percentile (Median)',
        showlegend=True
    ))

    # Theoretical Best Case (+1 bet every hand)
    best_case_y = params.bankroll + (x_axis * params.bet * 1.0)
    fig_traj.add_trace(go.Scatter(
        x=x_axis, y=best_case_y, mode='lines', 
        line=dict(color='gold', width=2, dash='dash'), 
        name='Theoretical Best (+1 win/hand)',
        showlegend=True
    ))
    
    # Theoretical Worst Case (-1 bet every hand)
    worst_case_y = params.bankroll + (x_axis * params.bet * -1.0)
    ruin_indices = np.where(worst_case_y < params.bet)[0]
    if len(ruin_indices) > 0:
        worst_case_y[ruin_indices[0]:] = worst_case_y[ruin_indices[0]]
        
    fig_traj.add_trace(go.Scatter(
        x=x_axis, y=worst_case_y, mode='lines', 
        line=dict(color='red', width=2, dash='dash'), 
        name='Theoretical Worst (-1 loss/hand)',
        showlegend=True
    ))
    
    fig_traj.update_layout(
        xaxis_title="Hands Played",
        yaxis_title="Bankroll ($)",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig_traj, use_container_width=True)

with col_chart2:
    st.subheader("📉 Time-to-Ruin Distribution")
    st.markdown("Histogram showing how long it takes to go broke (for those who do).")
    
    ttl_hours = (result.ttl_hands[result.ruined] * params.minutes_per_hand) / 60.0
    
    if len(ttl_hours) > 0:
        fig_hist = px.histogram(
            x=ttl_hours, 
            nbins=50,
            labels={'x': 'Hours to Ruin', 'y': 'Number of Players'},
            color_discrete_sequence=['#ef553b'] # Plotly Red
        )
        fig_hist.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Nobody went broke! Try increasing simulations or decreasing starting bankroll.")

st.divider()

# -----------------------------------------------------------------------------
# Dashboard Row 3: Survival Curve & Timetable
# -----------------------------------------------------------------------------
col_chart3, col_chart4 = st.columns(2)

with col_chart3:
    st.subheader("🛡️ Survival Curve")
    st.markdown("The probability of keeping your bankroll alive over time.")
    
    survival_hours = (result.survival_hands * params.minutes_per_hand) / 60.0
    
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(
        x=survival_hours,
        y=result.survival_prob * 100,
        mode='lines',
        line=dict(color='#ab63fa', width=3), # Plotly Purple
        name="Survival %",
        fill='tozeroy',
        fillcolor='rgba(171, 99, 250, 0.2)'
    ))
    
    fig_surv.update_layout(
        xaxis_title="Hours Played",
        yaxis_title="Survival Probability (%)",
        yaxis_range=[0, 105],
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig_surv, use_container_width=True)

with col_chart4:
    st.subheader("⏱️ Ruin Probability by Play Duration")
    st.markdown("Quick reference table for specific session lengths.")
    
    ruin_data = []
    for hours in [1, 2, 4, 8, 12, 24, 48]:
        hand_target = int(round(hours * hands_per_hour))
        if hand_target <= 0 or hand_target > params.max_hands: 
            continue
        
        # P(ruined by hand_target)
        ruined_by = np.mean(result.ruined & (result.ttl_hands <= hand_target))
        ruin_data.append({
            "Session Length": f"{hours} Hours", 
            "Hands Played": f"{hand_target:,}", 
            "Probability of Ruin": f"{ruined_by * 100:.2f}%"
        })
    
    if ruin_data:
        df_ruin = pd.DataFrame(ruin_data)
        st.dataframe(
            df_ruin, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Probability of Ruin": st.column_config.TextColumn(
                    "Probability of Ruin",
                    help="Chance your bankroll is empty before this time is up."
                )
            }
        )
