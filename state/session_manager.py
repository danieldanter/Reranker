# state/session_manager.py
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import os

import streamlit as st
from utils.azure_llm_client import AzureLLMClient, PDFProcessor


def _sanitize_filename(name: str) -> str:
    """Make a string safe for use as a Windows filename."""
    # Replace illegal characters \ / : * ? " < > |
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    # Strip leading/trailing spaces and dots
    name = name.strip(" .")
    # Fallback if empty
    return name or "questions"


class SessionManager:
    """Manages test questions and evaluation sessions"""

    def __init__(self, data_dir: str = "data"):
        # Helpful debug: where are we actually running?
        print(f"DEBUG(SessionManager): CWD={Path.cwd().absolute()}")

        self.data_dir = Path(data_dir)
        # Ensure parent dirs too (parents=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.questions_dir = self.data_dir / "questions"
        self.results_dir = self.data_dir / "results"
        self.questions_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"DEBUG(SessionManager): questions_dir={self.questions_dir.absolute()}")
        print(f"DEBUG(SessionManager): results_dir={self.results_dir.absolute()}")

        self.llm_client = AzureLLMClient()
        self.pdf_processor = PDFProcessor()

    def generate_questions_from_pdf(self, pdf_file, num_questions: int = 5) -> List[Dict]:
        """Generate Q&A pairs from uploaded PDF"""
        full_text = self.pdf_processor.extract_full_text(pdf_file)
        if not full_text:
            return []

        qa_pairs = self.llm_client.generate_questions_simple(full_text, num_questions)

        formatted_questions = []
        for i, qa in enumerate(qa_pairs, 1):
            formatted_questions.append({
                "id": f"Q{i}",
                "question": qa.get("question", ""),
                "ground_truth": qa.get("answer", ""),
                "type": qa.get("type", "factual"),
                "difficulty": qa.get("difficulty", "medium"),
                "status": "pending",
                "source": "auto-generated",
            })
        return formatted_questions

    # ------------------- Saving & Loading -------------------

    def save_questions(self, questions: List[Dict], name: str):
        """Save a set of test questions"""
        safe_name = _sanitize_filename(name)
        filepath = self.questions_dir / f"{safe_name}.json"

        print(f"DEBUG(save_questions): trying to save {len(questions)} questions")
        print(f"DEBUG(save_questions): safe_name='{safe_name}'")
        print(f"DEBUG(save_questions): path={filepath.absolute()}")

        try:
            # Ensure dir exists at time of save
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)
            print(f"DEBUG(save_questions): SUCCESS -> {filepath.absolute()}")
        except Exception as e:
            print(f"DEBUG(save_questions): FAILED -> {e}")
            raise

    def load_questions(self, name: str) -> List[Dict]:
        """Load a set of test questions"""
        safe_name = _sanitize_filename(name)
        filepath = self.questions_dir / f"{safe_name}.json"

        print(f"DEBUG(load_questions): loading '{safe_name}' from {filepath.absolute()}")

        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"DEBUG(load_questions): loaded {len(data)} questions")
                return data
            except Exception as e:
                print(f"DEBUG(load_questions): FAILED -> {e}")
                return []
        else:
            print("DEBUG(load_questions): file does not exist")
        return []

    def list_question_sets(self) -> List[str]:
        """List all available question sets (without .json)"""
        try:
            sets = sorted([f.stem for f in self.questions_dir.glob("*.json")])
            print(f"DEBUG(list_question_sets): found {len(sets)} sets in {self.questions_dir.absolute()}")
            return sets
        except Exception as e:
            print(f"DEBUG(list_question_sets): FAILED -> {e}")
            return []

    def save_results(self, results: Dict, session_name: Optional[str] = None):
        """Save evaluation results"""
        if not session_name:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        safe_name = _sanitize_filename(session_name)
        filepath = self.results_dir / f"{safe_name}.json"

        print(f"DEBUG(save_results): saving to {filepath.absolute()}")

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print("DEBUG(save_results): SUCCESS")
        except Exception as e:
            print(f"DEBUG(save_results): FAILED -> {e}")
            raise

    def load_results(self, session_name: str) -> Dict:
        """Load evaluation results"""
        safe_name = _sanitize_filename(session_name)
        filepath = self.results_dir / f"{safe_name}.json"

        print(f"DEBUG(load_results): loading from {filepath.absolute()}")

        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print("DEBUG(load_results): SUCCESS")
                return data
            except Exception as e:
                print(f"DEBUG(load_results): FAILED -> {e}")
                return {}
        else:
            print("DEBUG(load_results): file does not exist")
        return {}

    def create_sample_questions(self) -> List[Dict]:
        """Create sample questions for testing"""
        return [
            {
                "id": "Q1",
                "question": "Who is the author of the Bachelor thesis on Sarcopenia?",
                "ground_truth": "Daniel Danter",
                "type": "factual",
                "difficulty": "easy",
            },
            {
                "id": "Q2",
                "question": "What is the main topic of the thesis?",
                "ground_truth": "Sarcopenia - age-related loss of skeletal muscle mass and function",
                "type": "factual",
                "difficulty": "easy",
            },
            {
                "id": "Q3",
                "question": "Which university was this thesis submitted to?",
                "ground_truth": "University of Salzburg",
                "type": "factual",
                "difficulty": "medium",
            },
        ]
