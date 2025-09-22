import pandas as pd
from typing import List, Dict
import asyncio
from datetime import datetime
import json
from utils.ragas_metrics import RAGASMetrics
from utils.api_caller import VectorServiceCaller
from utils.answer_generator import AnswerGenerator

class BatchEvaluator:
    def __init__(self):
        self.metrics_calc = RAGASMetrics()
        self.api_caller = VectorServiceCaller()
        self.answer_gen = AnswerGenerator()
        
    def is_refusal_answer(self, answer: str) -> bool:
        """
        Detect if answer is a refusal/miss in German or English
        """
        answer_lower = answer.lower()
        
        # English refusal patterns
        english_patterns = [
            "do not provide information",
            "do not contain information",
            "cannot answer",
            "can't answer",
            "unable to answer",
            "no information found",
            "could not find",
            "not found in the documents",
            "documents don't contain",
            "if you could provide more context"
        ]
        
        # German refusal patterns
        german_patterns = [
            "keine information",
            "nicht gefunden",
            "kann nicht beantworten",
            "keine antwort",
            "nicht in den dokumenten",
            "dokumente enthalten nicht",
            "nicht verfügbar",
            "keine angaben",
            "wenn sie mehr kontext"
        ]
        
        # Check for any refusal pattern
        for pattern in english_patterns + german_patterns:
            if pattern in answer_lower:
                return True
        
        return False
        
    # In batch_evaluator.py, update the evaluate_single_question method:

    def evaluate_single_question(self, question_data: Dict, folder_ids: List[str], unique_titles: List[str]) -> Dict:
        """Evaluate a single question through both systems"""
        
        question = question_data['question']
        ground_truth = question_data['ground_truth']
        
        # Fetch chunks from both systems
        results = self.api_caller.fetch_both_systems(
            query=question,
            folder_ids=folder_ids,
            unique_titles=unique_titles,
            top_k=10
        )
        
        evaluation = {
            'question_id': question_data['id'],
            'question': question,
            'ground_truth': ground_truth
        }
        
        # Process each system
        for system in ['original', 'reranked']:
            try:
                # Check if we have valid chunks
                if results[system].get('error'):
                    print(f"Error in {system} system: {results[system]['error']}")
                    continue
                    
                chunks = results[system].get('chunks', [])
                
                # Check if chunks is actually a list
                if not isinstance(chunks, list):
                    print(f"Warning: chunks for {system} is not a list, got {type(chunks)}")
                    continue
                    
                if not chunks:
                    print(f"No chunks returned for {system} system")
                    continue
                
                contexts = [chunk.get('content', '') for chunk in chunks if isinstance(chunk, dict)]
                
                if not contexts:
                    print(f"No valid contexts extracted from chunks for {system}")
                    continue
                
                # Generate answer
                answer = self.answer_gen.generate_answer(question, chunks)
                
                # Calculate metrics
                metrics = {
                    'faithfulness': self.metrics_calc.calculate_faithfulness(answer, contexts),
                    'answer_relevancy': self.metrics_calc.calculate_answer_relevancy(question, answer),
                    'context_precision': self.metrics_calc.calculate_context_precision(question, contexts),
                    'context_recall': self.metrics_calc.calculate_context_recall(ground_truth, contexts),
                    'answer_correctness': self.metrics_calc.calculate_answer_correctness(answer, ground_truth),
                    'response_time_ms': results[system].get('time_ms', 0),
                    'answer': answer,
                    'refused_to_answer': self.is_refusal_answer(answer)  # Use the new function
                }
                
                evaluation[f'{system}_metrics'] = metrics
                
            except Exception as e:
                print(f"Error processing {system} system: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return evaluation
    
    # Add these methods to your existing BatchEvaluator class in utils/batch_evaluator.py

def evaluate_single_question_separate(self, question_data: Dict, 
                                     original_config: Dict,
                                     reranked_config: Dict) -> Dict:
    """
    Evaluate a single question through both systems with separate configurations
    
    Args:
        question_data: Dict with question info
        original_config: Dict with 'folder_ids' and 'unique_titles' for original system
        reranked_config: Dict with 'folder_ids' and 'unique_titles' for reranked system
    """
    
    question = question_data['question']
    ground_truth = question_data['ground_truth']
    
    # Fetch chunks from both systems with separate configurations
    results = self.api_caller.fetch_both_systems_separate(
        query=question,
        original_config=original_config,
        reranked_config=reranked_config,
        top_k=10
    )
    
    evaluation = {
        'question_id': question_data['id'],
        'question': question,
        'ground_truth': ground_truth,
        'original_config': original_config,  # Store config for reference
        'reranked_config': reranked_config
    }
    
    # Process each system
    for system in ['original', 'reranked']:
        try:
            if results[system].get('error'):
                print(f"Error in {system} system: {results[system]['error']}")
                continue
                
            chunks = results[system].get('chunks', [])
            
            if not isinstance(chunks, list):
                print(f"Warning: chunks for {system} is not a list, got {type(chunks)}")
                continue
                
            if not chunks:
                print(f"No chunks returned for {system} system")
                continue
            
            contexts = [chunk.get('content', '') for chunk in chunks if isinstance(chunk, dict)]
            
            if not contexts:
                print(f"No valid contexts extracted from chunks for {system}")
                continue
            
            # Generate answer
            answer = self.answer_gen.generate_answer(question, chunks)
            
            # Calculate metrics
            metrics = {
                'faithfulness': self.metrics_calc.calculate_faithfulness(answer, contexts),
                'answer_relevancy': self.metrics_calc.calculate_answer_relevancy(question, answer),
                'context_precision': self.metrics_calc.calculate_context_precision(question, contexts),
                'context_recall': self.metrics_calc.calculate_context_recall(ground_truth, contexts),
                'answer_correctness': self.metrics_calc.calculate_answer_correctness(answer, ground_truth),
                'response_time_ms': results[system].get('time_ms', 0),
                'answer': answer,
                'refused_to_answer': self.is_refusal_answer(answer),
                'source_config': results[system].get('config', {})  # Include source config
            }
            
            evaluation[f'{system}_metrics'] = metrics
            
        except Exception as e:
            print(f"Error processing {system} system: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return evaluation

def batch_evaluate_separate(self, questions: List[Dict], 
                           original_config: Dict,
                           reranked_config: Dict) -> tuple:
    """
    Evaluate multiple questions with separate configurations for each system
    
    Args:
        questions: List of question dictionaries
        original_config: Configuration for original system
        reranked_config: Configuration for reranked system
    
    Returns:
        Tuple of (DataFrame with results, raw results list)
    """
    
    all_results = []
    
    print(f"\nStarting batch evaluation with separate configurations:")
    print(f"Original config: {original_config}")
    print(f"Reranked config: {reranked_config}")
    print(f"Evaluating {len(questions)} questions...\n")
    
    for i, question_data in enumerate(questions, 1):
        print(f"Evaluating question {i}/{len(questions)}: {question_data['id']}")
        
        try:
            result = self.evaluate_single_question_separate(
                question_data, 
                original_config,
                reranked_config
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error evaluating {question_data['id']}: {e}")
            continue
    
    # Create results dataframe
    df = self._create_results_dataframe(all_results)
    
    # Add configuration info to the results
    if df is not None and not df.empty:
        # You could add config columns to track which sources were used
        df['original_source'] = str(original_config)
        df['reranked_source'] = str(reranked_config)
    
    return df, all_results
    
    def batch_evaluate(self, questions: List[Dict], folder_ids: List[str], unique_titles: List[str]) -> pd.DataFrame:
        """Evaluate multiple questions and return aggregated results"""
        
        all_results = []
        
        for i, question_data in enumerate(questions, 1):
            print(f"Evaluating question {i}/{len(questions)}: {question_data['id']}")
            
            try:
                result = self.evaluate_single_question(question_data, folder_ids, unique_titles)
                all_results.append(result)
            except Exception as e:
                print(f"Error evaluating {question_data['id']}: {e}")
                continue
        
        # Calculate aggregate metrics
        df = self._create_results_dataframe(all_results)
        return df, all_results
    
    def _create_results_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Create a comprehensive results dataframe"""
        
        metrics_data = []
        
        for result in results:
            # Check if we have metrics for both systems
            for system in ['original', 'reranked']:
                metrics_key = f'{system}_metrics'
                if metrics_key not in result:
                    print(f"Warning: No {system} metrics for question {result.get('question_id', 'unknown')}")
                    continue
                    
                metrics = result[metrics_key]
                row = {
                    'question_id': result['question_id'],
                    'system': system,
                    'faithfulness': metrics.get('faithfulness', 0),
                    'answer_relevancy': metrics.get('answer_relevancy', 0),
                    'context_precision': metrics.get('context_precision', 0),
                    'context_recall': metrics.get('context_recall', 0),
                    'answer_correctness_standard': metrics.get('answer_correctness', {}).get('standard_correctness', 0),
                    'answer_correctness_advanced': metrics.get('answer_correctness', {}).get('advanced_correctness', 0),
                    'semantic_similarity': metrics.get('answer_correctness', {}).get('semantic_similarity', 0),
                    'response_time_ms': metrics.get('response_time_ms', 0),
                    'refused_to_answer': metrics.get('refused_to_answer', False)
                }
                metrics_data.append(row)
        
        if not metrics_data:
            print("No metrics data collected!")
            return pd.DataFrame()  # Return empty DataFrame
        
        return pd.DataFrame(metrics_data)
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Generate evaluation report with statistical analysis"""
        
        # Calculate mean metrics for each system
        original_metrics = df[df['system'] == 'original'].mean(numeric_only=True)
        reranked_metrics = df[df['system'] == 'reranked'].mean(numeric_only=True)
        
        # Calculate improvements (fix division by zero)
        improvements = {}
        for metric in original_metrics.index:
            if metric != 'response_time_ms':
                original_val = original_metrics[metric]
                reranked_val = reranked_metrics[metric]
                
                if original_val == 0:
                    if reranked_val == 0:
                        improvements[metric] = 0  # Both are zero
                    else:
                        improvements[metric] = float('inf')  # Infinite improvement
                else:
                    improvements[metric] = ((reranked_val - original_val) / original_val * 100)
        
        # Statistical significance (paired t-test)
        from scipy import stats
        significance = {}
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
            original_values = df[df['system'] == 'original'][metric].values
            reranked_values = df[df['system'] == 'reranked'][metric].values
            t_stat, p_value = stats.ttest_rel(reranked_values, original_values)
            significance[metric] = {'t_stat': t_stat, 'p_value': p_value, 'significant': p_value < 0.05}
        
        report = {
            'original_metrics': original_metrics.to_dict(),
            'reranked_metrics': reranked_metrics.to_dict(),
            'improvements': improvements,
            'statistical_significance': significance,
            'total_questions': len(df) // 2
        }
        
        if not df.empty:
            original_misses = df[(df['system'] == 'original') & (df['refused_to_answer'] == True)].shape[0]
            reranked_misses = df[(df['system'] == 'reranked') & (df['refused_to_answer'] == True)].shape[0]
            total_questions = len(df) // 2
            
            # Add to report
            report['miss_statistics'] = {
                'original_misses': original_misses,
                'original_miss_rate': (original_misses / total_questions * 100) if total_questions > 0 else 0,
                'reranked_misses': reranked_misses,
                'reranked_miss_rate': (reranked_misses / total_questions * 100) if total_questions > 0 else 0,
                'miss_reduction': original_misses - reranked_misses
            }
        
        return report