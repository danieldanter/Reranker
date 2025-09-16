# app.py
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="RAG Reranker Evaluation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0
if 'test_questions' not in st.session_state:
    st.session_state.test_questions = []
if 'results_cache' not in st.session_state:
    st.session_state.results_cache = {}

# Title
st.title("🔍 RAG Evaluation Dashboard")
st.markdown("---")

# Import main comparison component
from components.unified_comparison import render_unified_comparison

# Render the main interface
render_unified_comparison()