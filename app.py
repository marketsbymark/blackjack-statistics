import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import simulation logic from the existing script
from bankroll_sim import (
    EV_LABEL,
    RULE_SET,
    SimParams,
    ruin_by_hours,
    simulate,
    wilson_interval,
)

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
    bankroll = st.number_input(
        "Starting bankroll ($)",
        min_value=float(bet),
        value=max(500.0, float(bet)),
        step=50.0,
        help="Must cover at least one bet.",
    )
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
    seed = st.number_input("Random seed", min_value=0, value=20260424, step=1, format="%d")
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
sidebar_params = SimParams(
    bet=float(bet),
    bankroll=float(bankroll),
    minutes_per_hand=float(minutes_per_hand),
    sims=int(sims),
    max_hands=int(max_hands),
    seed=int(seed),
    n_trajectories=int(n_trajectories)
)

if run_sim or "result" not in st.session_state:
    with st.spinner("Rolling the dice (running simulation)..."):
        params = sidebar_params

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

if params != sidebar_params:
    st.warning("Sidebar inputs have changed. Results below are from the previous run; press **Run Simulation** to update.")

# Calculate top-level stats
ruin_k = int(result.ruined.sum())
ruin_rate = ruin_k / params.sims
ruin_lo, ruin_hi = wilson_interval(ruin_k, params.sims)
ev_ci = 1.96 * result.ev_se_units
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
    help=(
        f"Bankroll fell below one bet within the {params.max_hands:,}-hand session. "
        f"95% Wilson CI: {ruin_lo * 100:.1f}% to {ruin_hi * 100:.1f}%."
    ),
)
col2.metric(
    label="Realized EV / Round",
    value=f"{ev * 100:+.2f}% ± {ev_ci * 100:.2f}%",
    help=(
        f"{EV_LABEL}, per initial bet, with 95% CI (SE clustered by session). "
        "Near ruin the player cannot fund every split/double, so this is not the "
        "unconstrained house edge."
    ),
)
col3.metric(
    label="Expected Loss per Hour",
    value=f"${ev_dollar_loss_per_hour:.2f}",
    help=f"Realized EV × ${params.bet}/hand × {hands_per_hour:.1f} hands/hour.",
)
col4.metric(
    label="Mean Ending Bankroll",
    value=f"${mean_ending_bankroll:,.2f}",
    delta=round(mean_net_result, 2),
    help="Average final bankroll across every simulation, including ruined players. Delta is vs the starting bankroll."
)

st.caption(
    f"Engine: {RULE_SET}. EV/SD measured from {result.hands_played_total:,} simulated rounds. "
    f"Ruin 95% CI: {ruin_lo * 100:.1f}%–{ruin_hi * 100:.1f}% ({params.sims:,} sessions)."
)

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
    x=["Split", "Double", "DAS", "Blackjack", "Push", "Player Bust", "Dealer Bust*", ">1 Bet Risked"],
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
st.caption("*Dealer only draws when the player still has a live hand, so this understates the dealer's raw bust rate.")

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
    st.markdown(
        "Paths of the first simulated sessions (sessions are independent, so this is an unbiased sample). "
        "The median and 10–90% band are computed over **all** simulations."
    )

    fig_traj = go.Figure()

    n_points = result.sample_paths.shape[1]
    x_axis = np.arange(n_points)

    # One trace with NaN separators renders far faster than one trace per path.
    x_all = np.tile(np.append(x_axis, np.nan), n_samples)
    y_all = np.hstack([result.sample_paths, np.full((n_samples, 1), np.nan)]).ravel()
    fig_traj.add_trace(go.Scattergl(
        x=x_all,
        y=y_all,
        mode='lines',
        line=dict(width=1, color='rgba(0, 204, 150, 0.35)'),
        name='Sample paths',
        hoverinfo='skip',
        connectgaps=False,
    ))

    band_x = result.survival_hands
    fig_traj.add_trace(go.Scatter(
        x=band_x, y=result.band_p90, mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ))
    fig_traj.add_trace(go.Scatter(
        x=band_x, y=result.band_p10, mode='lines',
        line=dict(width=0), fill='tonexty', fillcolor='rgba(255,255,255,0.15)',
        name='10–90% (all sims)',
    ))
    fig_traj.add_trace(go.Scatter(
        x=band_x, y=result.band_p50, mode='lines',
        line=dict(color='white', width=3),
        name='Median (all sims)',
    ))

    fig_traj.add_hline(
        y=params.bet,
        line_dash="dash",
        line_color="red",
        annotation_text="Ruin Line",
        annotation_position="bottom right"
    )

    observed_min = min(float(np.nanmin(result.sample_paths)), float(np.min(result.band_p10)))
    observed_max = max(float(np.nanmax(result.sample_paths)), float(np.max(result.band_p90)))
    y_span = max(observed_max - observed_min, params.bet * 4.0)
    y_padding = max(params.bet * 2.0, y_span * 0.08)
    y_min = min(observed_min, params.bet) - y_padding
    y_max = observed_max + y_padding

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
    st.markdown(
        f"Hours until ruin for the {ruin_k:,} sessions that went broke. "
        f"The other {params.sims - ruin_k:,} survived the full session and are not shown."
    )

    ttl_ruin_hours = result.ttl_hands[result.ruined] * params.minutes_per_hand / 60.0

    if len(ttl_ruin_hours) > 0:
        fig_hist = px.histogram(
            x=ttl_ruin_hours,
            nbins=50,
            labels={'x': 'Hours Until Ruin', 'y': 'Number of Sessions'},
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
        st.info("No simulated session went broke within the session length.")

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
    
    ruin_data = [
        {
            "Session Length": f"{row['hours']:g} Hours",
            "Hands Played": f"{row['hands']:,}",
            "Probability of Ruin": f"{row['p_ruin'] * 100:.2f}%",
            "95% CI": f"{row['ci_low'] * 100:.1f}% – {row['ci_high'] * 100:.1f}%",
        }
        for row in ruin_by_hours(params, result)
    ]

    df_ruin = pd.DataFrame(ruin_data)
    st.dataframe(
        df_ruin,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probability of Ruin": st.column_config.TextColumn(
                "Probability of Ruin",
                help="Chance your bankroll is below one bet by this point. Last row is the full simulated session."
            )
        }
    )
