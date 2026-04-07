import os
import json
import pandas as pd
from typing import Dict, Any, List
from env import AutoCleanEnv
from task import generate_task
from evaluator import evaluate_cleanliness


class AutoCleanAgent:
    def __init__(self):
        self.env = AutoCleanEnv()
        self.system_prompt = self._load_prompt('system.txt')
        self.cleaning_prompt = self._load_prompt('cleaning.txt')
        
    def _load_prompt(self, filename: str) -> str:
        try:
            path = os.path.join(os.path.dirname(__file__), 'prompts', filename)
            with open(path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"⚠️  Prompt file {filename} not found, using fallback")
            return ""
        except Exception as e:
            print(f"⚠️  Failed to load prompt {filename}: {str(e)}")
            return ""
            
    def _decide_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Decide next best action using heuristic/LLM logic"""
        try:
            metrics = observation.get('metrics', {})
            schema = observation.get('schema', {})
            
            if metrics.get('duplicate_ratio', 0) > 0.01:
                return {"type": "remove_duplicates", "params": {}}
                
            if metrics.get('missing_ratio', 0) > 0.05:
                df = observation.get('state')
                if df is None:
                    df = observation.get('dataset')
                if df is not None:
                    missing_cols = df.columns[df.isna().any()].tolist()
                    if missing_cols:
                        return {"type": "fill_missing", "params": {"column": missing_cols[0]}}
                    
            if metrics.get('type_consistency', 1.0) < 0.95:
                return {"type": "fix_types", "params": {}}
                
            if metrics.get('outlier_ratio', 0) > 0.02:
                numeric_cols = [col for col, dtype in schema.items() if dtype == 'numeric']
                if numeric_cols:
                    return {"type": "remove_outliers", "params": {"column": numeric_cols[0]}}
            
            return None
        except Exception as e:
            print(f"⚠️  Action decision failed: {str(e)}")
            return None
        
    def run(self, dataset: pd.DataFrame = None, max_steps: int = 50) -> Dict[str, Any]:
        """Run complete cleaning agent loop"""
        if dataset is None:
            dataset = generate_task()
            
        observation = self.env.reset(dataset)
        done = False
        
        while not done and self.env.current_step < max_steps:
            action = self._decide_action(observation)
            
            if action is None:
                break
                
            observation, reward, done, info = self.env.step(action)
            print(f"Step {self.env.current_step}: {action['type']} | Score: {reward:.4f}")
            
        final_report = self._generate_final_report()
        return final_report
        
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive cleaning report"""
        return {
            "success": self.env.reward >= 0.95,
            "final_score": self.env.reward,
            "initial_score": self.env.dirty_metrics['total_score'],
            "improvement": self.env.reward - self.env.dirty_metrics['total_score'],
            "steps_taken": self.env.current_step,
            "history": self.env.history,
            "final_metrics": self.env._calculate_metrics(self.env.state),
            "raw_dataset": self.env.raw_dataset,
            "cleaned_dataset": self.env.state,
            "versions": self.env.versions
        }


if __name__ == "__main__":
    try:
        agent = AutoCleanAgent()
        report = agent.run()
        
        print("\n✅ Cleaning Complete!")
        print(f"Initial Score: {report['initial_score']:.4f}")
        print(f"Final Score:   {report['final_score']:.4f}")
        print(f"Improvement:   {report['improvement']:.4f}")
        print(f"Steps Taken:   {report['steps_taken']}")
        print(f"Success:       {report['success']}")
        
        report['cleaned_dataset'].to_csv('cleaned_dataset.csv', index=False)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
