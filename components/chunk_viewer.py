# components/chunk_viewer.py
import streamlit as st
from typing import List, Dict, Optional

def calculate_chunk_movements(original_chunks: List[Dict], reranked_chunks: List[Dict]) -> tuple:
    """Calculate movement statistics between original and reranked results"""
    
    # Create mapping by uniqueTitle and chunkNr
    def get_chunk_id(chunk):
        return f"{chunk.get('uniqueTitle', '')}_{chunk.get('chunkNr', '')}"
    
    movements = {}
    stats = {
        'matching': 0,
        'new_in_top10': 0,
        'dropped_from_top10': 0,
        'biggest_jump': None,
        'biggest_drop': None,
        'from_outside_top10': []
    }
    
    # Map original chunks by ID (these are top 10 without reranking)
    original_ids = {get_chunk_id(chunk): i+1 for i, chunk in enumerate(original_chunks)}
    
    # Process reranked chunks
    for new_pos, chunk in enumerate(reranked_chunks, 1):
        chunk_id = get_chunk_id(chunk)
        original_pos = chunk.get('original_position', None)
        
        if chunk_id in original_ids:
            # This chunk was in original top 10
            actual_original_pos = original_ids[chunk_id]
            movement = actual_original_pos - new_pos
            movements[chunk_id] = {
                'original_pos': actual_original_pos,
                'new_pos': new_pos,
                'movement': movement,
                'status': 'moved'
            }
            stats['matching'] += 1
            
            # Track biggest movers
            if movement > 0 and (not stats['biggest_jump'] or movement > stats['biggest_jump']['movement']):
                stats['biggest_jump'] = {
                    'chunk_id': chunk_id,
                    'movement': movement,
                    'from': actual_original_pos,
                    'to': new_pos
                }
            elif movement < 0 and (not stats['biggest_drop'] or abs(movement) > abs(stats['biggest_drop']['movement'])):
                stats['biggest_drop'] = {
                    'chunk_id': chunk_id,
                    'movement': movement,
                    'from': actual_original_pos,
                    'to': new_pos
                }
        else:
            # This chunk came from outside original top 10
            movements[chunk_id] = {
                'original_pos': original_pos,  # This is position from the 30 candidates
                'new_pos': new_pos,
                'movement': original_pos - new_pos if original_pos else None,
                'status': 'new_from_deep'
            }
            stats['new_in_top10'] += 1
            
            if original_pos and original_pos > 10:
                stats['from_outside_top10'].append({
                    'chunk_id': chunk_id,
                    'from': original_pos,
                    'to': new_pos,
                    'jump': original_pos - new_pos
                })
    
    # Check which original chunks got dropped
    for chunk_id in original_ids:
        if chunk_id not in movements:
            movements[chunk_id] = {
                'original_pos': original_ids[chunk_id],
                'new_pos': None,
                'movement': None,
                'status': 'dropped'
            }
            stats['dropped_from_top10'] += 1
    
    return movements, stats

def get_movement_indicator(movement_info: Dict) -> str:
    """Get emoji and text for movement"""
    if movement_info['status'] == 'new_from_deep':
        orig = movement_info.get('original_pos', '?')
        if orig and orig > 10:
            jump = movement_info.get('movement', 0)
            return f"🚀 From #{orig} (↑{jump})"
        else:
            return "🆕 New"
    elif movement_info['status'] == 'dropped':
        return "❌ Dropped"
    elif movement_info['movement'] > 0:
        return f"⬆️ {movement_info['movement']}"
    elif movement_info['movement'] < 0:
        return f"⬇️ {abs(movement_info['movement'])}"
    else:
        return "➡️ Same"

def render_chunk_viewer(chunks: List[Dict], system_name: str, movements: Dict = None, selected_chunk: str = None):
    """Render chunks with movement indicators and highlighting"""
    
    if not chunks:
        st.warning("No chunks retrieved")
        return
    
    show_all = st.checkbox(f"Show all {len(chunks)} chunks", key=f"show_all_{system_name}")
    chunks_to_show = chunks if show_all else chunks[:10]
    
    for i, chunk in enumerate(chunks_to_show, 1):
        chunk_id = f"{chunk.get('uniqueTitle', '')}_{chunk.get('chunkNr', '')}"
        actual_chunk_number = chunk.get('chunkNr', '?')
        is_selected = (selected_chunk == chunk_id)
        
        # Apply highlighting if selected
        container_style = "background-color: #fffacd; padding: 10px; border-radius: 8px; margin: 5px 0;" if is_selected else ""
        
        with st.container():
            if is_selected:
                st.markdown(f'<div style="{container_style}">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 4, 1])
            
            with col1:
                # Click to select/highlight
                if st.button("🔍", key=f"select_{system_name}_{i}", 
                            help="Click to highlight matching chunk"):
                    if st.session_state.get('selected_chunk') == chunk_id:
                        st.session_state.selected_chunk = None
                    else:
                        st.session_state.selected_chunk = chunk_id
                    st.rerun()
            
            with col2:
                # Show actual chunk number prominently
                with st.expander(f"**Chunk {actual_chunk_number}** - Score: {chunk.get('score', 0):.4f}"):
                    st.markdown(f"**Document:** {chunk.get('title', 'N/A')}")
                    st.markdown(f"**Position in results:** #{i}")
                    
                    # Show original position if reranked
                    if chunk.get('was_reranked') and chunk.get('original_position'):
                        st.info(f"Was at position #{chunk['original_position']} before reranking (of 30 candidates)")
                    
                    st.markdown("**Content preview:**")
                    st.text(chunk.get('content', 'No content')[:500])
            
            with col3:
                # Different display logic for original vs reranked
                if movements and chunk_id in movements:
                    movement_info = movements[chunk_id]
                    
                    if system_name == "original":
                        # For original system: just show kept/dropped status
                        if movement_info['status'] == 'dropped':
                            st.error("❌ Dropped")
                        else:
                            st.success("✓ Kept")
                    
                    else:  # reranked system
                        # For reranked system with new color scheme
                        indicator = get_movement_indicator(movement_info)
                        
                        # Green for new chunks
                        if movement_info['status'] == 'new_from_deep':
                            st.success(indicator)
                        
                        # Yellow/Warning for upward movement or same position
                        elif movement_info.get('movement', 0) >= 0:
                            st.warning(indicator)
                        
                        # Blue/Info for downward movement
                        else:
                            st.info(indicator)
            
            if is_selected:
                st.markdown('</div>', unsafe_allow_html=True)