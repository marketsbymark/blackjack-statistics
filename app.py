import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import simulation logic from the existing script
from bankroll_sim import MAX_ROUND_NET_UNITS, RULE_SET, SimParams, simulate

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
card-level Monte Carlo simulation estimating how long your bankroll will last under realistic casino rules.

**Rules Modeled:** 6-deck shoe, H17, DAS, blackjack pays 3:2, no surrender/insurance,
basic strategy, flat betting, split to 4 hands, split aces receive one card only.
""")

# -----------------------------------------------------------------------------
# Sidebar: User Inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    
    bet = st.number_input("Dollars per hand", min_value=1.0, value=25.0, step=5.0)
    bankroll = st.number_input("Starting bankroll ($)", min_value=1.0, value=500.0, step=50.0)
    minutes_per_hand = st.number_input("Minutes per hand", min_value=0.1, value=0.75, step=0.1)
    session_hours = st.number_input("Session length (hours)", min_value=0.25, value=4.0, step=0.25)
    max_hands = max(1, int(round((session_hours * 60.0) / minutes_per_hand)))
    st.caption(f"Simulating about {max_hands:,} hands per session.")
    
    st.divider()
    
    sims = st.selectbox(
        "Number of simulations",
        options=[100, 500, 1000, 5000, 10000],
        index=2,
        help="The full card engine is much slower than the old PMF model; increase this for smoother curves."
    )
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
        result = simulate(params)
        
        # Cache results in session state so changing tabs/interacting doesn't re-run
        st.session_state["result"] = result
        st.session_state["params"] = params

# -----------------------------------------------------------------------------
# Extract Cache for Rendering
# -----------------------------------------------------------------------------
result = st.session_state["result"]
params = st.session_state["params"]
ev = result.ev_units
sd = result.sd_units

# Calculate top-level stats
ruin_rate = float(result.ruined.mean())
dollar_loss_per_hand = -ev * params.bet
hands_per_hour = 60.0 / params.minutes_per_hand
ev_dollar_loss_per_hour = dollar_loss_per_hand * hands_per_hour
survivors = result.ending_bankroll[~result.ruined]
mean_survivor_bankroll = float(np.mean(survivors)) if survivors.size else 0.0
mean_ending_bankroll = float(np.mean(result.ending_bankroll))
mean_net_result = mean_ending_bankroll - params.bankroll

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
    label="Mean Ending Bankroll",
    value=f"${mean_ending_bankroll:,.2f}",
    delta=f"${mean_net_result:,.2f} vs start",
    delta_color="normal" if mean_net_result >= 0 else "inverse",
    help="Average final bankroll across every simulation, including ruined players."
)

st.caption(f"Engine: {RULE_SET}. EV/SD are measured from {result.hands_played_total:,} simulated rounds.")

surv_col1, surv_col2 = st.columns(2)
surv_col1.metric(
    label="Survivors",
    value=f"{survivors.size:,} / {params.sims:,}",
    help="Simulations that did not hit ruin before max hands."
)
surv_col2.metric(
    label="Mean Survivor Bankroll",
    value=f"${mean_survivor_bankroll:,.2f}",
    help="Conditional average among survivors only. This is intentionally separated because it excludes ruined players."
)

col5, col6, col7, col8 = st.columns(4)
col5.metric(
    label="Rounds With Split",
    value=f"{result.split_rate * 100:.2f}%",
    help="Percent of initial rounds that included at least one split."
)
col6.metric(
    label="Rounds With Double",
    value=f"{result.double_rate * 100:.2f}%",
    help="Percent of initial rounds that included at least one double down."
)
col7.metric(
    label="Rounds With DAS",
    value=f"{result.das_rate * 100:.2f}%",
    help="Percent of initial rounds that included at least one double after split."
)
col8.metric(
    label="Avg / Max Exposure",
    value=f"{result.avg_exposure_units:.2f}x / {result.max_exposure_units:.0f}x",
    help="Average and maximum base-bet exposure required during a round."
)

st.divider()

event_fig = go.Figure(go.Bar(
    x=["Split", "Double", "DAS", "Blackjack", "Push", "Player Bust", "Dealer Bust", ">1 Bet Risked"],
    y=[
        result.split_rate * 100,
        result.double_rate * 100,
        result.das_rate * 100,
        result.blackjack_rate * 100,
        result.push_rate * 100,
        result.player_bust_rate * 100,
        result.dealer_bust_rate * 100,
        result.multi_bet_rate * 100,
    ],
    marker_color=["#00cc96", "#636efa", "#ab63fa", "#facc15", "#19d3f3", "#ef553b", "#ffa15a", "#b6e880"],
))
event_fig.update_layout(
    title="Round Event Probabilities",
    yaxis_title="Rounds with event (%)",
    height=320,
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
)
st.plotly_chart(event_fig, use_container_width=True)

action_fig = go.Figure(go.Bar(
    x=["Split actions", "Double actions", "DAS actions"],
    y=[
        result.split_action_rate * 100,
        result.double_action_rate * 100,
        result.das_action_rate * 100,
    ],
    marker_color=["#00cc96", "#636efa", "#ab63fa"],
))
action_fig.update_layout(
    title="Action Counts per Initial Round",
    yaxis_title="Actions per 100 initial rounds",
    height=280,
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
)
st.plotly_chart(action_fig, use_container_width=True)

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
    observed_min = float(np.nanmin(result.sample_paths))
    observed_max = float(np.nanmax(result.sample_paths))
    y_span = max(observed_max - observed_min, params.bet * 4.0)
    y_padding = max(params.bet * 2.0, y_span * 0.08)
    y_min = min(observed_min, params.bet) - y_padding
    y_max = observed_max + y_padding
    
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

    # Theoretical extreme case: split to 4 hands, double all 4, win every doubled hand.
    best_case_y = params.bankroll + (x_axis * params.bet * MAX_ROUND_NET_UNITS)
    fig_traj.add_trace(go.Scatter(
        x=x_axis, y=best_case_y, mode='lines', 
        line=dict(color='gold', width=2, dash='dash'), 
        name=f'Theoretical Max (+{MAX_ROUND_NET_UNITS:.0f} units/round)',
        showlegend=True
    ))
    
    # Theoretical extreme loss mirrors the max funded split/double exposure.
    worst_case_y = params.bankroll + (x_axis * params.bet * -MAX_ROUND_NET_UNITS)
    ruin_indices = np.where(worst_case_y < params.bet)[0]
    if len(ruin_indices) > 0:
        worst_case_y[ruin_indices[0]:] = worst_case_y[ruin_indices[0]]
        
    fig_traj.add_trace(go.Scatter(
        x=x_axis, y=worst_case_y, mode='lines', 
        line=dict(color='red', width=2, dash='dash'), 
        name=f'Theoretical Min (-{MAX_ROUND_NET_UNITS:.0f} units/round)',
        showlegend=True
    ))
    
    fig_traj.update_layout(
        xaxis_title="Hands Played",
        yaxis_title="Bankroll ($)",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
        ),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            range=[y_min, y_max],
        )
    )
    st.plotly_chart(fig_traj, use_container_width=True)

with col_chart2:
    st.subheader("📉 Time-to-Ruin Distribution")
    st.markdown("Histogram showing how many hands it takes to go broke (for those who do).")
    
    ttl_ruin_hands = result.ttl_hands[result.ruined]
    
    if len(ttl_ruin_hands) > 0:
        fig_hist = px.histogram(
            x=ttl_ruin_hands,
            nbins=50,
            labels={'x': 'Hands Until Ruin', 'y': 'Number of Players'},
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
