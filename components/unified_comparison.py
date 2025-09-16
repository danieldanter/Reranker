# components/unified_comparison.py
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List
import json
from datetime import datetime

from components.question_navigator import render_question_navigator
from components.chunk_viewer import render_chunk_viewer, calculate_chunk_movements
from state.session_manager import SessionManager
from utils.answer_generator import AnswerGenerator
from utils.api_caller import VectorServiceCaller
from utils.batch_evaluator import BatchEvaluator

def render_unified_comparison():
    """Main unified comparison interface"""
    
    # Initialize components
    session_mgr = SessionManager()
    api_caller = VectorServiceCaller()
    
    # Ensure some session defaults
    st.session_state.setdefault("test_questions", [])
    st.session_state.setdefault("current_question_idx", 0)
    st.session_state.setdefault("results_cache", {})
    st.session_state.setdefault("last_generated_questions", [])
    st.session_state.setdefault("generated_questions", [])
    st.session_state.setdefault("generated_questions_name", "")
    st.session_state.setdefault("save_debug", {})

    # ── Session management section (top-level expander) ──────────────────────────
    with st.expander("📁 Session Management", expanded=False):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Load Questions", "Generate from PDF", "Manual Entry", "Tab4", "RAGAS Evaluation"])
        
        # --- Tab 1: Load Questions ---
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                question_sets = session_mgr.list_question_sets()
                if question_sets:
                    selected_set = st.selectbox("Load Question Set", [""] + question_sets)
                    if selected_set and st.button("Load"):
                        questions = session_mgr.load_questions(selected_set)
                        st.session_state.test_questions = questions
                        st.session_state.current_question_idx = 0
                        st.success(f"Loaded {len(questions)} questions")
            with col2:
                if st.button("Load Sample Questions"):
                    st.session_state.test_questions = session_mgr.create_sample_questions()
                    st.session_state.current_question_idx = 0
                    st.success("Loaded sample questions")
        
        # --- Tab 2: Generate from PDF ---
        # --- Tab 2: Generate from PDF ---
        with tab2:
            st.markdown("### 🤖 Generate Q&A from PDF")
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            
            # Helper: show preview + actions if we already have generated questions in session
            def render_preview_and_actions():
                questions_to_use = st.session_state.get("last_generated_questions", [])
                default_save = st.session_state.get("last_save_name")
                
                if not questions_to_use:
                    return
                
                st.markdown("### 📋 Preview Generated Questions")
                preview_count = min(10, len(questions_to_use))
                for q in questions_to_use[:preview_count]:
                    p_col1, p_col2 = st.columns([3, 1])
                    with p_col1:
                        st.markdown(f"**{q['id']}:** {q['question']}")
                        st.markdown(f"**A:** {q['ground_truth']}")
                    with p_col2:
                        st.caption(f"Type: {q.get('type','')}")
                        st.caption(f"Level: {q.get('difficulty','')}")
                    st.markdown("---")
                if len(questions_to_use) > preview_count:
                    st.info(f"... and {len(questions_to_use) - preview_count} more questions")

                # Action buttons (always read from session state)
                col_add, col_replace, col_save_use = st.columns(3)

                with col_add:
                    if st.button("➕ Add to Current", use_container_width=True, key="btn_add_to_current"):
                        st.session_state.test_questions.extend(questions_to_use)
                        st.success("Added to current questions!")
                        st.rerun()

                with col_replace:
                    if st.button("🔄 Replace All", use_container_width=True, key="btn_replace_all"):
                        st.session_state.test_questions = questions_to_use
                        st.session_state.current_question_idx = 0
                        st.success("Replaced all questions!")
                        st.rerun()

                with col_save_use:
                    # Persist the save name in session so it survives across reruns
                    current_default = (
                        st.session_state.get("generated_questions_name")
                        or default_save
                        or "qa_set"
                    )
                    save_name_input = st.text_input(
                        "Save as:",
                        value=current_default,
                        key="save_name_generated",
                    )
                    st.session_state.generated_questions_name = save_name_input

                    if st.button("💾 Save & Use", use_container_width=True, type="primary", key="btn_save_and_use"):
                        questions_to_save = st.session_state.get("last_generated_questions", [])
                        save_name = st.session_state.get("generated_questions_name", "backup")

                        print("DEBUG(UI): Save & Use clicked.")
                        print(f"DEBUG(UI): About to save {len(questions_to_save)} questions as '{save_name}'")

                        if questions_to_save:
                            # Persist to disk
                            session_mgr.save_questions(questions_to_save, save_name)
                            # Load into active session
                            st.session_state.test_questions = questions_to_save
                            st.session_state.current_question_idx = 0
                            # Refresh saved sets (if displayed elsewhere)
                            if hasattr(session_mgr, "list_question_sets"):
                                st.session_state.saved_question_sets = session_mgr.list_question_sets()
                            # Clear transient generated cache if you want to avoid accidental re-saves
                            del st.session_state["last_generated_questions"]
                            st.success(f"✅ Saved and loaded {len(questions_to_save)} questions as '{save_name}'")
                            st.rerun()
                        else:
                            st.error("No questions to save!")

            if uploaded_file:
                # Get file info
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
                st.info(f"📄 File: {uploaded_file.name} ({file_size:.1f} MB)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    num_questions = st.number_input(
                        "Number of questions to generate",
                        min_value=5, max_value=50, value=20, step=5,
                        help="Generate between 5 and 50 questions from the PDF"
                    )
                with col2:
                    process_mode = st.selectbox(
                        "Processing mode",
                        ["Full Document", "Smart Sampling"],
                        help="Full Document: Process entire PDF (slower but comprehensive)\nSmart Sampling: Sample from document (faster)"
                    )
                with col3:
                    question_focus = st.selectbox(
                        "Question focus",
                        ["Mixed", "Factual Only", "Analytical", "Details"],
                        help="Type of questions to generate"
                    )

                # Advanced options
                show_advanced = st.checkbox("Show Advanced Options")
                include_page_refs = True
                avoid_duplicates = True
                focus_sections = []
                if show_advanced:
                    st.markdown("##### ⚙️ Advanced Options")
                    a_col1, a_col2 = st.columns(2)
                    with a_col1:
                        include_page_refs = st.checkbox("Include page references in answers", value=True)
                        avoid_duplicates = st.checkbox("Avoid duplicate questions", value=True)
                    with a_col2:
                        focus_sections = st.multiselect(
                            "Focus on sections (optional)",
                            ["Introduction", "Methods", "Results", "Discussion", "Conclusion"],
                            help="Leave empty to cover entire document"
                        )
                
                if st.button(f"🚀 Generate {num_questions} Questions", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                        # Step 1: Extract text
                        status_text.text("Step 1/3: Extracting PDF text...")
                        progress_bar.progress(0.2)
                        full_text = session_mgr.pdf_processor.extract_full_text(uploaded_file)
                        
                        if not full_text:
                            st.error("Could not extract text from PDF")
                        else:
                            text_length = len(full_text)
                            st.success(f"✅ Extracted {text_length:,} characters from PDF")
                            
                            # Step 2: Generate questions
                            status_text.text(f"Step 2/3: Generating {num_questions} questions...")
                            progress_bar.progress(0.5)
                            # If you later support the UI options, pass them here as kwargs
                            new_questions = session_mgr.llm_client.generate_questions_simple(
                                full_text, num_questions
                            )
                            
                            progress_bar.progress(0.9)
                            
                            if new_questions:
                                status_text.text("Step 3/3: Formatting results...")
                                progress_bar.progress(1.0)
                                
                                # Format questions
                                formatted_questions = []
                                for i, qa in enumerate(new_questions, 1):
                                    formatted_questions.append({
                                        "id": f"Q{i}",
                                        "question": qa.get("question", ""),
                                        "ground_truth": qa.get("answer", ""),
                                        "type": qa.get("type", "faktisch"),
                                        "difficulty": qa.get("difficulty", "mittel"),
                                        "status": "pending",
                                        "source": f"auto-generated from {uploaded_file.name}"
                                    })

                                # Compute a default save name immediately (timestamped)
                                default_save = (
                                    f"{uploaded_file.name.replace('.pdf', '')}"
                                    f"_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                )
                                
                                # CRITICAL: store in session state so it survives reruns & button clicks
                                st.session_state.last_generated_questions = formatted_questions
                                st.session_state.generated_questions = formatted_questions  # (if used elsewhere)
                                st.session_state.last_save_name = default_save
                                st.session_state.generated_questions_name = default_save

                                # Save debug info about save path
                                questions_dir = getattr(session_mgr, "questions_dir", None)
                                save_debug = {
                                    "questions_dir": str(questions_dir.absolute()) if questions_dir else "(unknown)",
                                    "default_save": default_save,
                                    "expected_path": (
                                        str((questions_dir / f"{default_save}.json").absolute())
                                        if questions_dir else "(unknown)"
                                    ),
                                    "count": len(formatted_questions),
                                }
                                st.session_state.save_debug = save_debug
                                
                                # Terminal debug prints
                                print(f"DEBUG(UI): Prepared {len(formatted_questions)} questions")
                                if questions_dir:
                                    print(f"DEBUG(UI): questions_dir={questions_dir.absolute()}")
                                print(f"DEBUG(UI): default_save={default_save}")
                                if questions_dir:
                                    print(f"DEBUG(UI): expected_path={(questions_dir / f'{default_save}.json').absolute()}")

                                st.success(f"✅ Generated {len(formatted_questions)} unique questions!")

                                # Metrics
                                s_col1, s_col2, s_col3 = st.columns(3)
                                with s_col1:
                                    st.metric("Questions Generated", len(formatted_questions))
                                with s_col2:
                                    types = [q['type'] for q in formatted_questions]
                                    st.metric("Question Types", len(set(types)))
                                with s_col3:
                                    avg_len = sum(len(q['question']) for q in formatted_questions) / len(formatted_questions)
                                    st.metric("Avg Question Length", f"{avg_len:.0f} chars")

                                # Render preview + actions from session state
                                render_preview_and_actions()
                            else:
                                st.error("Failed to generate questions")
                    
                    progress_bar.empty()
                    status_text.empty()

                # If we already have generated questions from a previous run, show them + actions
                if st.session_state.get("last_generated_questions"):
                    st.markdown("---")
                    render_preview_and_actions()
                else:
                    # Optional gentle hint
                    st.caption("Generate questions to preview and save them here.")


        # --- Tab 3: Manual Entry ---
        with tab3:
            if st.button("Add New Question"):
                st.session_state.show_add_question = True

        # Add new tab for batch evaluation
        # Update the batch evaluation section in unified_comparison.py
        with tab5:  # Add as new tab
            st.markdown("### 🚀 RAGAS Batch Evaluation")
            
            if st.session_state.get("test_questions"):
                st.info(f"Ready to evaluate {len(st.session_state.test_questions)} questions")
                
                # Pull current search configuration from the existing input fields (set earlier in the app)
                current_folder_id = st.session_state.get('folder_id_input', 'dd29b9bd-31aa-4db1-8208-767f52332735')
                current_unique_title = st.session_state.get('unique_title_input', '')

                # Show current configuration
                st.markdown("#### Current Search Configuration:")
                col_cfg_1, col_cfg_2 = st.columns(2)
                with col_cfg_1:
                    st.text(f"Folder ID: {current_folder_id if current_folder_id else 'Not set'}")
                with col_cfg_2:
                    st.text(f"Document: {current_unique_title if current_unique_title else 'All documents in folder'}")

                col1, col2 = st.columns(2)
                with col1:
                    num_to_evaluate = st.number_input(
                        "Number of questions to evaluate",
                        min_value=1,
                        max_value=len(st.session_state.test_questions),
                        value=min(30, len(st.session_state.test_questions))
                    )

                with col2:
                    if st.button("🎯 Run RAGAS Evaluation", type="primary"):
                        # Validate configuration
                        if not current_folder_id:
                            st.error("Please set a Folder ID in the Search Configuration section")
                        else:
                            evaluator = BatchEvaluator()
                            with st.spinner(f"Evaluating {num_to_evaluate} questions... This may take a few minutes."):
                                questions_subset = st.session_state.test_questions[:num_to_evaluate]

                                # Use values from the input fields
                                df, raw_results = evaluator.batch_evaluate(
                                    questions_subset,
                                    folder_ids=[current_folder_id] if current_folder_id else [],
                                    unique_titles=[current_unique_title] if current_unique_title else []
                                )

                                report = evaluator.generate_report(df)

                                # Store results
                                st.session_state.ragas_results = {
                                    'df': df,
                                    'raw_results': raw_results,
                                    'report': report
                                }

                # Display results if available
                if 'ragas_results' in st.session_state:
                    report = st.session_state.ragas_results['report']
                    df = st.session_state.ragas_results['df']

                    st.markdown("### 📊 RAGAS Evaluation Results")

                    # Overall scores
                    col_sum_1, col_sum_2 = st.columns(2)
                    with col_sum_1:
                        original_score = np.mean([
                            report['original_metrics']['faithfulness'],
                            report['original_metrics']['answer_relevancy'],
                            report['original_metrics']['context_precision'],
                            report['original_metrics']['context_recall']
                        ])
                        st.metric("Original System RAGAS Score", f"{original_score:.3f}")

                    with col_sum_2:
                        reranked_score = np.mean([
                            report['reranked_metrics']['faithfulness'],
                            report['reranked_metrics']['answer_relevancy'],
                            report['reranked_metrics']['context_precision'],
                            report['reranked_metrics']['context_recall']
                        ])
                        improvement = ((reranked_score - original_score) / original_score) * 100 if original_score != 0 else 0.0
                        st.metric("Reranked System RAGAS Score", f"{reranked_score:.3f}", f"+{improvement:.1f}%")

                    # Detailed metrics table
                    st.markdown("### Detailed Metrics Comparison")

                    metrics_comparison = pd.DataFrame({
                        'Metric': [
                            'Faithfulness', 
                            'Answer Relevancy', 
                            'Context Precision', 
                            'Context Recall', 
                            'Answer Correctness (Standard)',
                            'Answer Correctness (Advanced)'
                        ],
                        'Original': [
                            report['original_metrics']['faithfulness'],
                            report['original_metrics']['answer_relevancy'],
                            report['original_metrics']['context_precision'],
                            report['original_metrics']['context_recall'],
                            report['original_metrics']['answer_correctness_standard'],
                            report['original_metrics']['answer_correctness_advanced']
                        ],
                        'Reranked': [
                            report['reranked_metrics']['faithfulness'],
                            report['reranked_metrics']['answer_relevancy'],
                            report['reranked_metrics']['context_precision'],
                            report['reranked_metrics']['context_recall'],
                            report['reranked_metrics']['answer_correctness_standard'],
                            report['reranked_metrics']['answer_correctness_advanced']
                        ],
                        'Improvement (%)': [
                            report['improvements']['faithfulness'],
                            report['improvements']['answer_relevancy'],
                            report['improvements']['context_precision'],
                            report['improvements']['context_recall'],
                            report['improvements']['answer_correctness_standard'],
                            report['improvements']['answer_correctness_advanced']
                        ]
                    })

                    st.dataframe(
                        metrics_comparison.style.format({
                            'Original': '{:.3f}',
                            'Reranked': '{:.3f}',
                            'Improvement (%)': '{:+.1f}'
                        })
                    )

                    # Miss Statistics
                    if 'miss_statistics' in report:
                        st.markdown("### 🎯 Answer Coverage Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Original System Misses",
                                f"{report['miss_statistics']['original_misses']}/{report['total_questions']}",
                                f"{report['miss_statistics']['original_miss_rate']:.1f}% miss rate"
                            )
                        
                        with col2:
                            st.metric(
                                "Reranked System Misses",
                                f"{report['miss_statistics']['reranked_misses']}/{report['total_questions']}",
                                f"{report['miss_statistics']['reranked_miss_rate']:.1f}% miss rate"
                            )
                        
                        with col3:
                            reduction = report['miss_statistics']['miss_reduction']
                            if reduction > 0:
                                st.metric("Miss Reduction", f"-{reduction}", "Improvement", delta_color="normal")
                            else:
                                st.metric("Miss Reduction", f"{reduction}", "No improvement", delta_color="off")

                    # Advanced vs Standard Correctness Comparison
                    st.markdown("### 📊 Answer Correctness Analysis")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Standard Correctness** (Exact Match)")
                        standard_orig = report['original_metrics'].get('answer_correctness_standard', 0)
                        standard_rerank = report['reranked_metrics'].get('answer_correctness_standard', 0)
                        st.metric("Original", f"{standard_orig:.3f}")
                        st.metric("Reranked", f"{standard_rerank:.3f}", f"{((standard_rerank - standard_orig) / standard_orig * 100):.1f}%")

                    with col2:
                        st.markdown("**Advanced Correctness** (Semantic Match)")
                        advanced_orig = report['original_metrics'].get('answer_correctness_advanced', 0)
                        advanced_rerank = report['reranked_metrics'].get('answer_correctness_advanced', 0)
                        st.metric("Original", f"{advanced_orig:.3f}")
                        st.metric("Reranked", f"{advanced_rerank:.3f}", f"{((advanced_rerank - advanced_orig) / advanced_orig * 100):.1f}%")

                     # NEW: Question-by-Question Results
                    st.markdown("### 📋 Question-Level Results")
                    
                    raw_results = st.session_state.ragas_results.get('raw_results', [])
                    
                    for result in raw_results:
                        st.markdown(f"#### Question {result['question_id']}: {result['question'][:50]}...")
                        with st.container():
                            st.markdown(f"**Ground Truth:** {result['ground_truth']}")
                            
                            # Create comparison table for this question
                            question_metrics = []
                            
                            for system in ['original', 'reranked']:
                                if f'{system}_metrics' in result:
                                    metrics = result[f'{system}_metrics']
                                    question_metrics.append({
                                        'System': system.capitalize(),
                                        'Faithfulness': metrics.get('faithfulness', 0),
                                        'Answer Relevancy': metrics.get('answer_relevancy', 0),
                                        'Context Precision': metrics.get('context_precision', 0),
                                        'Context Recall': metrics.get('context_recall', 0),
                                        'Answer Correctness (Standard)': metrics.get('answer_correctness', {}).get('standard_correctness', 0),
                                        'Answer Correctness (Advanced)': metrics.get('answer_correctness', {}).get('advanced_correctness', 0),
                                        'All Facts Present': '✓' if metrics.get('answer_correctness', {}).get('all_facts_present', False) else '✗',
                                        'Refused to Answer': '✓' if metrics.get('refused_to_answer', False) else '✗'
                                    })
                                    
                                    # Show the actual answers
                                    st.markdown(f"**{system.capitalize()} Answer:**")
                                    st.info(metrics.get('answer', 'No answer generated'))
                            
                            if question_metrics:
                                df_question = pd.DataFrame(question_metrics)
                                st.dataframe(df_question.style.format({
                                    'Faithfulness': '{:.3f}',
                                    'Answer Relevancy': '{:.3f}',
                                    'Context Precision': '{:.3f}',
                                    'Context Recall': '{:.3f}',
                                    'Answer Correctness (Standard)': '{:.3f}',
                                    'Answer Correctness (Advanced)': '{:.3f}'
                                }))

                    # Statistical significance
                    st.markdown("### Statistical Significance")
                    for metric, sig in report['statistical_significance'].items():
                        if sig['significant']:
                            st.success(f"✓ {metric}: p-value = {sig['p_value']:.4f} (significant improvement)")
                        else:
                            st.info(f"○ {metric}: p-value = {sig['p_value']:.4f} (not significant)")

                    # Export options
                    st.markdown("### Export Results")
                    exp_col1, exp_col2, exp_col3 = st.columns(3)

                    with exp_col1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "ragas_evaluation.csv",
                            "text/csv"
                        )

                    with exp_col2:
                        json_str = json.dumps(st.session_state.ragas_results['raw_results'], indent=2)
                        st.download_button(
                            "📥 Download JSON",
                            json_str,
                            "ragas_evaluation.json",
                            "application/json"
                        )
            else:
                st.warning("Please load questions first")

    


    # ── Add new question dialog ────────────────────────────────────────────────
    if st.session_state.get('show_add_question', False):
        with st.form("add_question_form"):
            st.subheader("Add New Question")
            q_text = st.text_input("Question")
            q_truth = st.text_area("Ground Truth Answer")
            q_type = st.selectbox("Type", ["factual", "analytical", "comparative"])
            if st.form_submit_button("Add"):
                new_q = {
                    "id": f"Q{len(st.session_state.test_questions) + 1}",
                    "question": q_text,
                    "ground_truth": q_truth,
                    "type": q_type,
                    "status": "pending"
                }
                st.session_state.test_questions.append(new_q)
                st.session_state.show_add_question = False
                st.rerun()
    # ── Search Configuration Section (before Question Navigator) ──
    st.markdown("### 🔧 Search Configuration")
    col1, col2 = st.columns(2)

    with col1:
        folder_id_input = st.text_input(
            "Folder ID(s) - comma separated",
            value="dd29b9bd-31aa-4db1-8208-767f52332735",
            help="Enter one or more folder IDs, separated by commas",
            key="folder_ids_config"
        )
        
    with col2:
        unique_title_input = st.text_input(
            "Document(s) - comma separated",
            value="dd29b9bd-31aa-4db1-8208-767f52332735_Bachelorarbeit Sarkopenie.pdf",
            help="Enter specific document IDs, or leave empty to search entire folder",
            key="unique_titles_config"
        )

    # Parse the inputs
    folder_ids = [f.strip() for f in folder_id_input.split(',') if f.strip()] if folder_id_input else []
    unique_titles = [u.strip() for u in unique_title_input.split(',') if u.strip()] if unique_title_input else []

    st.markdown("---")
    # ── Question Navigator ────────────────────────────────────────────────────
    st.markdown("### Question Navigator")
    new_idx = render_question_navigator(
        st.session_state.test_questions,
        st.session_state.current_question_idx
    )
    if new_idx is not None:
        st.session_state.current_question_idx = new_idx
        st.rerun()
    
    # ── Main comparison view ──────────────────────────────────────────────────
    if st.session_state.test_questions:
        current_q = st.session_state.test_questions[st.session_state.current_question_idx]
        st.markdown("---")
        st.markdown(f"### Current Question: {current_q['id']}")
        st.markdown(f"**Question:** {current_q['question']}")
        st.markdown(f"**Ground Truth:** {current_q['ground_truth']}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            run_current = st.button("🚀 Run Current", type="primary")
        with col2:
            run_all = st.button("📊 Run All")
        with col3:
            st.button("⬅️ Previous")
        with col4:
            st.button("➡️ Next")
        
        if run_current:
            # Use the configured values from above
            if not folder_ids and not unique_titles:
                st.error("Please provide either a Folder ID or Document ID")
            else:
                with st.spinner("Fetching results..."):
                    results = api_caller.fetch_both_systems(
                        query=current_q['question'],
                        folder_ids=folder_ids,
                        unique_titles=unique_titles,
                        top_k=10
                    )
                    cache_key = f"{current_q['id']}_results"
                    st.session_state.results_cache[cache_key] = results
            current_q['status'] = 'completed'
        # Initialize answer generator
        answer_gen = AnswerGenerator()
        # Add this where results are displayed
        cache_key = f"{current_q['id']}_results"
        if cache_key in st.session_state.results_cache:
            results = st.session_state.results_cache[cache_key]
            
            # Initialize selected chunk state
            if 'selected_chunk' not in st.session_state:
                st.session_state.selected_chunk = None
            
            # Calculate movements
            original_chunks = results["original"]["chunks"]
            reranked_chunks = results["reranked"]["chunks"]
            movements, stats = calculate_chunk_movements(original_chunks, reranked_chunks)

            # NEW: Generate and Compare Answers
            st.markdown("### 🤖 AI Answer Comparison")
            st.markdown("Comparing answers generated from original vs reranked chunks:")
            
            # Generate answers button
            if st.button("🔮 Generate Answers from Both Chunk Sets", type="primary"):
                with st.spinner("Generating answers..."):
                    # Generate answer from original chunks
                    original_answer = answer_gen.generate_answer(
                        current_q['question'], 
                        original_chunks[:10]  # Use top 10
                    )
                    
                    # Generate answer from reranked chunks
                    reranked_answer = answer_gen.generate_answer(
                        current_q['question'],
                        reranked_chunks[:10]  # Use top 10
                    )
                    
                    # Store in session state
                    st.session_state[f"{cache_key}_answers"] = {
                        'original': original_answer,
                        'reranked': reranked_answer
                    }
            
            # Display answers if they exist
            answers_key = f"{cache_key}_answers"
            if answers_key in st.session_state:
                answers = st.session_state[answers_key]
                
                col_orig_answer, col_rerank_answer = st.columns(2)
                
                with col_orig_answer:
                    st.markdown("#### 🔵 Answer from Original Chunks")
                    with st.container():
                        st.markdown(answers['original'])
                
                with col_rerank_answer:
                    st.markdown("#### 🟢 Answer from Reranked Chunks")
                    with st.container():
                        st.markdown(answers['reranked'])
                
                # Quality comparison section
                with st.expander("📊 Answer Quality Analysis"):
                    st.markdown("""
                    **Compare the answers on:**
                    - ✅ Completeness - Which answer is more comprehensive?
                    - ✅ Accuracy - Which answer better matches the ground truth?
                    - ✅ Citations - Which answer has better source attribution?
                    - ✅ Relevance - Which answer better addresses the question?
                    """)
                    
                    # Show ground truth for comparison
                    st.info(f"**Ground Truth:** {current_q['ground_truth']}")
            
            st.markdown("---")


            
            # Display statistics summary box
            st.markdown("### 📊 Reranking Impact Analysis")
            
            # Main stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Kept in Top 10", f"{stats['matching']}/10",
                        help="Chunks that stayed in top 10")
            
            with col2:
                st.metric("Pulled from Deep", stats['new_in_top10'],
                        help="New chunks from positions >10")
            
            with col3:
                st.metric("Dropped Out", stats['dropped_from_top10'],
                        help="Original top 10 chunks that got pushed out")
            
            with col4:
                if stats['from_outside_top10']:
                    best_find = max(stats['from_outside_top10'], key=lambda x: x['jump'])
                    st.metric("Best Find", f"#{best_find['from']} → #{best_find['to']}",
                            help=f"Biggest jump: moved up {best_find['jump']} positions!")
            
            # Show notable movements
            if stats['from_outside_top10']:
                with st.expander("🚀 Chunks Retrieved from Deep Positions"):
                    for item in sorted(stats['from_outside_top10'], key=lambda x: x['jump'], reverse=True):
                        st.write(f"• Position #{item['from']} → #{item['to']} (jumped {item['jump']} positions)")
            
            # Display the chunks with movements
            st.markdown("---")
            col_original, col_reranked = st.columns(2)
            
            with col_original:
                st.markdown("#### 🔵 ORIGINAL SYSTEM")
                if results["original"]["error"]:
                    st.error(f"Error: {results['original']['error']}")
                else:
                    st.metric("Response Time", f"{results['original']['time_ms']:.0f}ms")
                    render_chunk_viewer(original_chunks, "original", movements, 
                                    st.session_state.get('selected_chunk'))
            
            with col_reranked:
                st.markdown("#### 🟢 RERANKED SYSTEM")
                if results["reranked"]["error"]:
                    st.error(f"Error: {results['reranked']['error']}")
                else:
                    st.metric("Response Time", f"{results['reranked']['time_ms']:.0f}ms")
                    render_chunk_viewer(reranked_chunks, "reranked", movements,
                                    st.session_state.get('selected_chunk'))
