# components/question_navigator.py
import streamlit as st
from typing import List, Dict, Optional

# In question_navigator.py, modify the button handling:
def render_question_navigator(questions: List[Dict], current_idx: int) -> Optional[int]:
    if not questions:
        st.info("No questions loaded. Please add questions or load a test set.")
        return None
    
    # Create columns for question pills
    cols_per_row = 6
    num_rows = (len(questions) + cols_per_row - 1) // cols_per_row
    
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            q_idx = row * cols_per_row + col_idx
            if q_idx < len(questions):
                with cols[col_idx]:
                    # Use a callback instead of returning index
                    if st.button(
                        f"Q{q_idx+1}",
                        key=f"nav_q_{q_idx}",
                        type="primary" if q_idx == current_idx else "secondary",
                        use_container_width=True,
                        on_click=lambda idx=q_idx: st.session_state.update({'current_question_idx': idx})
                    ):
                        pass  # The callback handles the update
    
    return None  # No need to return index anymore