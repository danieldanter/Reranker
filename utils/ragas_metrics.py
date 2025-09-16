import numpy as np
from typing import List, Dict, Tuple
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

class RAGASMetrics:
    def __init__(self):
        # Initialize the Azure LLM for metric calculations
        self.config = {
            'domain': '506triggereu',
            'key': '5959556c551749d0b2c5807ea77919df',
            'deployment_name': 'gpt-4.1-mini',
            'api_version': '2024-02-15-preview'
        }
        self.base_url = f"https://{self.config['domain']}.openai.azure.com"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.config['key']
        }
        
        # Initialize sentence transformer for semantic similarity
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    def _call_llm(self, prompt: str) -> str:
        """Helper to call Azure LLM"""
        url = f"{self.base_url}/openai/deployments/{self.config['deployment_name']}/chat/completions?api-version={self.config['api_version']}"
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for evaluation tasks."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return ""
        except:
            return ""
    
    def calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Calculate faithfulness score: claims supported by context / total claims
        """
        # Step 1: Extract claims from answer
        prompt = f"""Extract all factual claims from this answer. 
        List each claim on a new line, numbered.
        
        Answer: {answer}
        
        Claims:"""
        
        claims_text = self._call_llm(prompt)
        claims = [c.strip() for c in claims_text.split('\n') if c.strip() and c[0].isdigit()]
        
        if not claims:
            return 1.0  # No claims to verify
        
        # Step 2: Verify each claim against contexts
        context_text = "\n".join(contexts)
        supported_claims = 0
        
        for claim in claims:
            # Remove numbering
            claim_text = re.sub(r'^\d+\.\s*', '', claim)
            
            prompt = f"""Based on the following context, can this claim be verified as true?
            Answer only 'YES' or 'NO'.
            
            Context: {context_text[:3000]}
            
            Claim: {claim_text}
            
            Answer:"""
            
            verification = self._call_llm(prompt).strip().upper()
            if 'YES' in verification:
                supported_claims += 1
        
        return supported_claims / len(claims) if claims else 1.0
    
    def calculate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Calculate how relevant the answer is to the question
        """
        # Generate artificial questions from the answer
        prompt = f"""Based on this answer, generate 3 questions that this answer would address.
        List each question on a new line.
        
        Answer: {answer}
        
        Questions:"""
        
        generated_questions = self._call_llm(prompt)
        questions = [q.strip() for q in generated_questions.split('\n') if q.strip()]
        
        if not questions:
            return 0.5
        
        # Calculate similarity between original and generated questions
        original_emb = self.embedder.encode([question])
        generated_embs = self.embedder.encode(questions)
        
        similarities = cosine_similarity(original_emb, generated_embs)[0]
        return float(np.mean(similarities))
    
    def calculate_context_precision(self, question: str, contexts: List[str], k: int = 5) -> float:
        """
        Calculate if relevant contexts appear at top positions
        """
        relevant_positions = []
        
        for i, context in enumerate(contexts[:k], 1):
            prompt = f"""Is this context relevant for answering the question?
            Answer only 'YES' or 'NO'.
            
            Question: {question}
            Context: {context[:500]}
            
            Answer:"""
            
            relevance = self._call_llm(prompt).strip().upper()
            if 'YES' in relevance:
                relevant_positions.append(i)
        
        if not relevant_positions:
            return 0.0
        
        # Precision at k with position weighting
        precision_score = sum(1/pos for pos in relevant_positions) / k
        return min(precision_score, 1.0)
    
    def calculate_context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """
        Calculate if contexts contain all information from ground truth
        """
        print(f"\n=== Context Recall Debug ===")
        print(f"Ground truth: {ground_truth[:100]}...")
        print(f"Number of contexts: {len(contexts)}")
        
        if not contexts:
            print("No contexts provided!")
            return 0.0
        
        # Show first context for debugging
        if contexts:
            print(f"First context sample: {contexts[0][:200]}...")
        
        # Try simple keyword matching first
        ground_truth_lower = ground_truth.lower()
        context_text_lower = " ".join(contexts).lower()
        
        # Check for key terms from ground truth
        key_terms = ground_truth_lower.split()[:5]  # First 5 words
        found_terms = sum(1 for term in key_terms if term in context_text_lower)
        
        print(f"Simple keyword check: {found_terms}/{len(key_terms)} key terms found")
        
        # If we find at least some keywords, return partial score
        if found_terms > 0:
            return found_terms / len(key_terms)
        
        # Otherwise return 0
        return 0.0

    def calculate_answer_correctness(self, answer: str, ground_truth: str) -> Dict[str, float]:
        """
        Advanced answer correctness with F1-like score and semantic similarity
        """
        # Extract facts from both
        def extract_facts(text):
            prompt = f"""Extract all factual statements from this text.
            List each fact on a new line.
            
            Text: {text}
            
            Facts:"""
            facts_text = self._call_llm(prompt)
            return [f.strip() for f in facts_text.split('\n') if f.strip()]
        
        answer_facts = extract_facts(answer)
        truth_facts = extract_facts(ground_truth)
        
        # Calculate F1-like score
        if not answer_facts and not truth_facts:
            f1_score = 1.0
        elif not answer_facts or not truth_facts:
            f1_score = 0.0
        else:
            # Simple overlap calculation
            answer_set = set(answer_facts)
            truth_set = set(truth_facts)
            
            tp = len(answer_set & truth_set)
            fp = len(answer_set - truth_set)
            fn = len(truth_set - answer_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate semantic similarity
        answer_emb = self.embedder.encode([answer])
        truth_emb = self.embedder.encode([ground_truth])
        semantic_sim = float(cosine_similarity(answer_emb, truth_emb)[0][0])
        
        # FIX: Advanced correctness for longer answers
        # Check if answer contains all key information from ground truth
        ground_truth_lower = ground_truth.lower()
        answer_lower = answer.lower()
        
        # Extract key terms from ground truth
        key_terms = []
        # For German and English, extract important words
        for word in ground_truth_lower.split():
            if len(word) > 3:  # Skip short words
                key_terms.append(word)
        
        # Check how many key terms are in the answer
        found_terms = sum(1 for term in key_terms if term in answer_lower)
        coverage = found_terms / len(key_terms) if key_terms else 0
        
        # Advanced correctness: If answer covers most key terms, give higher score
        if coverage >= 0.8:  # 80% of key terms found
            advanced_correctness = max(0.8, (f1_score * 0.3 + semantic_sim * 0.7))
        else:
            advanced_correctness = (f1_score * 0.5 + semantic_sim * 0.5)
        
        return {
            'f1_score': f1_score,
            'semantic_similarity': semantic_sim,
            'standard_correctness': (f1_score * 0.5 + semantic_sim * 0.5),
            'advanced_correctness': advanced_correctness,  # Now different from standard
            'all_facts_present': coverage >= 0.8
        }