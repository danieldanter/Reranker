import streamlit as st

def metrics_panel(metrics):
    with st.expander("Metrics", expanded=False):
        if not metrics:
            st.caption("No metrics yet.")
            return
        for k, v in metrics.items():
            st.metric(label=k, value=v)