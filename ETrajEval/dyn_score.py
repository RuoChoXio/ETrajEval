# dyn_score.py

import os
import re
import json
import math
import torch
import numpy as np
from typing import List, Dict, Any

from model import RewardModel
 
class DynamicEmotionalScorer:
    """
    Analyzes conversation transcripts to score emotional trajectories.
    It self-manages its prompt and parameter configurations.
    """
    def __init__(self, language: str):
        """
        Initializes the scorer.

        Args:
            language (str): The selected language key ('ch' or 'en').
        """
        self.language = language
        
        prompt_path = os.path.join("prompt", "sys_prompt.json")
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)['scoring_prompts']
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(f"FATAL: Scoring prompts could not be loaded from '{prompt_path}'. Details: {e}")
        
        self.main_template = prompts_data['main_template'][self.language]
        self.positive_eval = prompts_data['positive_eval'][self.language]
        self.negative_eval = prompts_data['negative_eval'][self.language]
        self.first_turn_context = prompts_data['first_turn_context'][self.language]
        self.descriptors = prompts_data['emotional_descriptors']

        params_path = os.path.join("prompt", "dynamic_params.json")
        try:
            with open(params_path, 'r', encoding='utf-8') as f:
                self.all_param_sets = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"FATAL: Dynamic parameters file not found at '{params_path}'")
            
        print(f"[*] DynamicEmotionalScorer initialized for language: '{language}'")

    def _get_dynamic_distribution_params(self, turn_number: int, params: Dict[str, float]) -> tuple[float, float]:
        """
        Calculates mu and sigma based on the provided parameter set.
        The decay starts from turn_number = 1 (which corresponds to the second subject turn).
        """
        n = float(turn_number)
        mu = params['mu_0'] * math.exp(-params['k_decay'] * (n - 1))
        variance = params['sigma_sq_final'] - \
                   (params['sigma_sq_final'] - params['sigma_sq_initial']) * math.exp(-params['r_decay'] * (n - 1))
        sigma = math.sqrt(max(variance, 1e-6))
        
        return mu, sigma

    def _map_score_to_emotional_descriptor(self, score: float) -> str:
        """
        Maps a raw logit score to a human-readable emotional descriptor
        using the externalized, bilingual configuration.
        """
        for descriptor_info in self.descriptors:
            if score < descriptor_info['threshold']:
                
                return descriptor_info[self.language]

    def _format_history(self, dialogue_history: List[Dict[str, str]], character_map: Dict[str, str]) -> str:
        """
        Formats the dialogue history into a string for the prompt.
        Handles the case for the very first turn.
        """
        if not dialogue_history: return self.first_turn_context
        formatted_string = ""
        for turn in dialogue_history:
            role_name = character_map.get(turn['role'], turn['role'])
            formatted_string += f"- {role_name}: {turn['content']}\n"
        
        return formatted_string.strip()
    
    def weighted_trajectory_score_corrected(self, turn_scores: List[float], param_set_key: str) -> float:
        """
        Recalculates the weighted trajectory score based on turn-skipping rules.

            - slow_40_turns: No turns are skipped.
            - medium_20_turns: Skips the score change between turns 21 and 22.
            - fast_10_turns: Skips changes between turns 11-12, 21-22, and 31-32.
        """
        if len(turn_scores) < 2:
            return 0.0

        skip_indices = set()
        if param_set_key == "medium_20_turns":
            skip_indices.add(21)
        elif param_set_key == "fast_10_turns":
            skip_indices.update([11, 21, 31])

        n = len(turn_scores)
        weighted_sum_of_diffs = 0.0
        valid_diff_count = 0

        for i in range(1, n):
            if i in skip_indices:
                continue

            score_curr = turn_scores[i]
            score_prev = turn_scores[i-1]
            diff = score_curr - score_prev
            weight = 1.0 - score_prev
            weighted_sum_of_diffs += diff * weight
            valid_diff_count += 1

        if valid_diff_count == 0:
            return 0.0
        
        return weighted_sum_of_diffs / valid_diff_count

def analyze_conversation_and_get_trajectory(
        self,
        conversation_log: List[Dict[str, str]],
        reward_model: RewardModel,
        k_samples: int,
        softmax_temperature: float,
        param_set_key: str,
        initial_mu: float
    ) -> Dict[str, Any]:
        """
        Main analysis pipeline.
        MODIFIED: Correctly handles the first turn without sampling and starts decay from the second turn.
        Removes batching for individual turn analysis.
        """
        base_params = self.all_param_sets.get(param_set_key)
        if not base_params:
            print(f"[!] Warning: param_set_key '{param_set_key}' not found. Using first available set.")
            base_params = next(iter(self.all_param_sets.values()))
        
        current_params = base_params.copy()
        current_params['mu_0'] = initial_mu

        reset_interval_match = re.search(r'(\d+)', param_set_key)
        reset_interval = int(reset_interval_match.group(1)) if reset_interval_match else 10
        
        final_turn_results = []
        if not conversation_log: return {"turn_scores": [], "trajectory_score": 0.0}

        subject_character_name = conversation_log[0]['role']
        assistant_character_name = conversation_log[1]['role'] if len(conversation_log) > 1 else "user"
        character_map = {subject_character_name: "user", assistant_character_name: "assistant"}
        
        subject_turns = [(i, turn) for i, turn in enumerate(conversation_log) if turn["role"] == subject_character_name]
        initial_turn_descriptor = "N/A (Initial turn)" if self.language == 'en' else "无 (初始轮次)"
        
        for original_index, current_turn in subject_turns:
            
            subject_turn_number = len(final_turn_results) + 1 # Determine turn number based on already processed turns

            history_context = conversation_log[:original_index]
            formatted_history = self._format_history(history_context, character_map)
            
            if current_turn["content"] == "":
                if self.language == 'en':
                    current_turn["content"] = "(User has no reply this turn)"
                elif self.language == 'zh':
                    current_turn["content"] = "（用户本轮没有回复）"

            batch_pos_prompts, batch_neg_prompts = [], []

            if subject_turn_number == 1:
                # For the first turn, do not sample. Use a fixed, neutral descriptor.
                descriptor = initial_turn_descriptor
                
                main_content = self.main_template.format(history=formatted_history, descriptor=descriptor, reply=current_turn["content"])
                for _ in range(k_samples):
                    batch_pos_prompts.append([{"role": "user", "content": main_content}, {"role": "assistant", "content": self.positive_eval}])
                    batch_neg_prompts.append([{"role": "user", "content": main_content}, {"role": "assistant", "content": self.negative_eval}])
            else:
                # For subsequent turns, apply the dynamic parameter decay and sample.
                effective_decay_step = ((subject_turn_number - 2) % reset_interval) + 1
                
                if subject_turn_number > 1 and effective_decay_step == 1:
                   print(f"    - Turn {subject_turn_number}: Scoring parameters have been reset.")
                
                mu, sigma = self._get_dynamic_distribution_params(effective_decay_step, current_params)
                prior_scores = torch.normal(mean=mu, std=sigma, size=(k_samples,))
                
                for score in prior_scores:
                    descriptor = self._map_score_to_emotional_descriptor(score.item())
                    main_content = self.main_template.format(history=formatted_history, descriptor=descriptor, reply=current_turn["content"])
                    batch_pos_prompts.append([{"role": "user", "content": main_content}, {"role": "assistant", "content": self.positive_eval}])
                    batch_neg_prompts.append([{"role": "user", "content": main_content}, {"role": "assistant", "content": self.negative_eval}])

            if not batch_pos_prompts: continue # Should not happen with k_samples > 0

            all_logits = reward_model.get_batch_scores(batch_pos_prompts + batch_neg_prompts)
            
            k_scores = []
            for k in range(k_samples):
                logit_pos_idx = k
                logit_neg_idx = len(batch_pos_prompts) + k # Corrected index for negative prompts
                
                logit_pos = all_logits[logit_pos_idx]
                logit_neg = all_logits[logit_neg_idx]
                
                scaled_pos, scaled_neg = logit_pos / softmax_temperature, logit_neg / softmax_temperature
                max_l = max(scaled_pos, scaled_neg)
                exp_pos, exp_neg = math.exp(scaled_pos - max_l), math.exp(scaled_neg - max_l)
                k_scores.append(exp_pos / (exp_pos + exp_neg))
            
            final_turn_results.append({
                "turn_index": original_index,
                "subject_turn_number": subject_turn_number,
                "average_sentiment_score": np.mean(k_scores),
                "sentiment_label": "positive" if np.mean(k_scores) > 0.5 else "negative",
            })
            # logits_processed += k_samples # This line is no longer needed as we process turns individually

        final_turn_results.sort(key=lambda x: x['turn_index'])
        
        unweighted_scores = [turn['average_sentiment_score'] for turn in final_turn_results]
        legacy_trajectory_score = (unweighted_scores[-1] - unweighted_scores[0]) / (len(unweighted_scores) - 1) if len(unweighted_scores) > 1 else 0.0

        return {"turn_scores": final_turn_results, "trajectory_score": legacy_trajectory_score}