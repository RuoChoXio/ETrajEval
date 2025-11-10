# run.py

import yaml
import os
import json
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
from collections import defaultdict
import argparse

def main():
    """Main function to run the chat simulation and scoring."""
    parser = argparse.ArgumentParser(description="Run chat simulation and scoring from a config file.")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the configuration YAML file.'
    )
    args = parser.parse_args()
    start_time = time.time()
    error_logs = []

    print(f"[*] Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[!] Error: Config file not found at '{args.config}'. Please check the path.")
        return

    exec_phases_config = config.get('execution_phases', {})
    run_generation = exec_phases_config.get('run_generation', False)
    run_scoring = exec_phases_config.get('run_scoring', False)

    if not run_generation and not run_scoring:
        print("[!] Both 'run_generation' and 'run_scoring' are disabled in the config. Exiting.")
        return

    language = config.get('language', 'en')
    if "visible_gpu_ids" in config and config["visible_gpu_ids"]:
        gpu_ids = [str(i) for i in config["visible_gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        print(f"[*] Setting CUDA_VISIBLE_DEVICES='{os.environ['CUDA_VISIBLE_DEVICES']}'")

    from model import BaseModel, LocalModel, APIModel, RewardModel
    from dataloader import load_structured_prompts
    from chat_env import ChatEnvironment
    from dyn_score import DynamicEmotionalScorer

    def create_model_from_config(model_config: Dict[str, Any]) -> BaseModel:
        model_type = model_config.get("type")
        if model_type == "local":
            return LocalModel(name=model_config['name'], path_or_name=model_config['path_or_name'], gpu_indices=model_config['gpu_indices'], generation_params=model_config.get('generation_params', {}))
        elif model_type == "api":
            api_conf = model_config['api_config']
            return APIModel(name=model_config['name'], model_name=model_config['model_name'], api_key=api_conf.get('api_key'), base_url=api_conf.get('base_url'), generation_params=model_config.get('generation_params', {}))
        else:
            raise ValueError(f"Unsupported model type '{model_type}' in config.")

    try:
        model1 = create_model_from_config(config['model1'])
        model2 = create_model_from_config(config['model2'])
    except (ValueError, KeyError) as e:
        print(f"[!] Error creating models from config: {e}")
        return

    prompts_data = load_structured_prompts(config['data']['prompt_file'])
    if not prompts_data:
        return

    # output_conf = config.get('output', {})
    output_conf = config.get('output') or {}
    model_output_folder_name = f"{model1.name}_vs_{model2.name}"

    # Path for Phase 1: Chat Generation Logs
    generation_base_dir = output_conf.get('generation_log_dir', '../DATA_ETEval_answer/chat_dial')
    generation_output_dir = os.path.join(generation_base_dir, model_output_folder_name)
    generation_output_file = os.path.join(generation_output_dir, f"chat_logs_{language}.json") # TODO!!

    # Path for Phase 2: Evaluation Results
    evaluation_base_dir = output_conf.get('evaluation_result_dir', '../DATA_ETEval_answer/eval_output')
    evaluation_output_dir = os.path.join(evaluation_base_dir, model_output_folder_name)
    evaluation_output_file = os.path.join(evaluation_output_dir, f"answer_dialogues_{language}.json")
    error_log_file = os.path.join(evaluation_output_dir, f"error_logs_{language}.json")

    conversations_to_process = []

    if run_generation:
        print("\n" + "="*25 + " PHASE 1: CHAT GENERATION " + "="*25)
        chat_conf = config['chat']
        chat_env = ChatEnvironment(model1=model1, model2=model2, language=language)
        max_concurrency = chat_conf.get('max_concurrent_chats', 1)
        
        print(f"[*] Starting generation with a concurrency of {max_concurrency} dialogues per batch.")
        
        with tqdm(total=len(prompts_data), desc="Overall Scenario Progress") as pbar:
            for i in range(0, len(prompts_data), max_concurrency):
                prompt_chunk = prompts_data[i:i + max_concurrency]
                
                try:
                    chunk_results = chat_env.run_parallel_chats(
                        prompt_scenarios=prompt_chunk,
                        max_turns=chat_conf['max_turns']
                    )
                    conversations_to_process.extend(chunk_results)
                except Exception as e:
                    print(f"\n[!!!] CRITICAL ERROR during batch starting at index {i}: {e}")
                    for scenario in prompt_chunk:
                        error_logs.append({"id": scenario['id'], "error_message": str(e), "original_prompt": scenario})
                    print(f"    - This batch of {len(prompt_chunk)} scenarios has been skipped.")
                
                pbar.update(len(prompt_chunk))
        
        # Use the determined path for saving
        os.makedirs(generation_output_dir, exist_ok=True)
        print(f"\n[*] All chat batches complete. Saving logs to: {generation_output_file}")
        with open(generation_output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations_to_process, f, ensure_ascii=False, indent=2)
        print("[+] Logs saved successfully!")
        
        if not run_scoring:
            print("\n[*] Phase 1 (generation) complete. Phase 2 (scoring) is disabled. Task finished.")
            total_execution_time = time.time() - start_time
            print(f"[*] Total execution time: {round(total_execution_time, 2)} seconds.")
            return

    if run_scoring:
        print("\n" + "="*25 + " PHASE 2: EMOTIONAL SCORING " + "="*25)
        scoring_conf = config.get("scoring")
        if not scoring_conf:
            print("[!] 'scoring' config not found, but 'run_scoring' is true. Skipping scoring phase.")
            return

        if not run_generation:
            # Load from a specific file if provided, otherwise use the (potentially customized) generation output path
            input_file_path = scoring_conf.get('input_chat_log_file') or generation_output_file
            
            print(f"[*] Attempting to load conversations for scoring from: {input_file_path}")
            try:
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    conversations_to_process = json.load(f)
                print(f"[+] Successfully loaded {len(conversations_to_process)} conversations.")
            except FileNotFoundError:
                print(f"[!!!] FATAL ERROR: Chat log file not found at '{input_file_path}'. Cannot proceed with scoring.")
                return
            except json.JSONDecodeError:
                print(f"[!!!] FATAL ERROR: Failed to decode JSON from '{input_file_path}'. Cannot proceed with scoring.")
                return

        if not conversations_to_process:
            print("[*] No successful conversations were available to score. Skipping scoring phase.")
            return

        try:
            reward_model = RewardModel(path=scoring_conf['reward_model_path'], device=scoring_conf['reward_model_device'])
            scorer = DynamicEmotionalScorer(language=language)
        except (KeyError, ValueError, RuntimeError) as e:
            print(f"[!] Error initializing scoring components: {e}"); return

        scored_results_data = []
        prompt_map = {p['id']: p for p in prompts_data}
        
        print(f"[*] Starting scoring for {len(conversations_to_process)} dialogues...")
        for conv_data in tqdm(conversations_to_process, desc="Scoring Conversations"):
            scenario_id = conv_data['id']
            prompt_scenario = prompt_map.get(scenario_id)
            if not prompt_scenario: continue

            analysis_results = scorer.analyze_conversation_and_get_trajectory(
                conversation_log=conv_data['conversations'], reward_model=reward_model,
                k_samples=scoring_conf['k_samples'], softmax_temperature=scoring_conf.get('softmax_temperature', 10),
                param_set_key=prompt_scenario.get("param_set_key", "medium_20_turns"),
                initial_mu=prompt_scenario.get("initial_mu", -2.0)
            )
            
            turn_scores = [turn['average_sentiment_score'] for turn in analysis_results['turn_scores']]
            # weighted_score = scorer.calculate_weighted_trajectory_score(turn_scores) if turn_scores else 0.0
            weighted_score = scorer.weighted_trajectory_score_corrected(turn_scores=turn_scores, param_set_key=prompt_scenario.get("param_set_key", "medium_20_turns")) if turn_scores else 0.0
            avg_sentiment = np.mean(turn_scores) if turn_scores else 0.0

            scored_results_data.append({
                "id": scenario_id, "psychological_theory": prompt_scenario.get("psychological_theory", "N/A"),
                "conversations": conv_data['conversations'], "scoring_results": analysis_results,
                "weighted_trajectory_score": weighted_score, "average_sentiment_score": avg_sentiment
            })
        
        total_execution_time = time.time() - start_time
        all_weighted_scores = [res['weighted_trajectory_score'] for res in scored_results_data]
        all_sentiment_scores = [res['average_sentiment_score'] for res in scored_results_data]
        theory_aggregator = defaultdict(lambda: {'weighted_scores': [], 'sentiment_scores': []})
        for res in scored_results_data:
            theory = res['psychological_theory']
            theory_aggregator[theory]['weighted_scores'].append(res['weighted_trajectory_score'])
            theory_aggregator[theory]['sentiment_scores'].append(res['average_sentiment_score'])
        
        theory_breakdown = {
            theory: {
                "average_weighted_trajectory_score": np.mean(data['weighted_scores']) if data['weighted_scores'] else 0.0,
                "average_sentiment_score": np.mean(data['sentiment_scores']) if data['sentiment_scores'] else 0.0,
                "scenario_count": len(data['weighted_scores'])
            } for theory, data in theory_aggregator.items()
        }

        final_output_data = {
            "metadata": {
                "model_1_name": model1.name, "model_2_name": model2.name,
                "total_execution_time_seconds": round(total_execution_time, 2),
                "total_scenarios_run": len(conversations_to_process),
                "total_scenarios_failed": len(error_logs),
                "overall_summary": {
                    "average_weighted_trajectory_score": round(np.mean(all_weighted_scores) if all_weighted_scores else 0.0, 6),
                    "overall_average_sentiment_score": round(np.mean(all_sentiment_scores) if all_sentiment_scores else 0.0, 6)
                },
                "theory_breakdown": theory_breakdown
            },
            "results": scored_results_data
        }

        # Use the determined path for saving evaluation results
        os.makedirs(evaluation_output_dir, exist_ok=True)
        print(f"\n[*] All scoring complete. Saving final evaluation results to: {evaluation_output_file}")
        with open(evaluation_output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output_data, f, ensure_ascii=False, indent=2)

        if error_logs:
            # Use the determined path for saving error logs
            error_file_path = os.path.join(evaluation_output_dir, f"error_logs_{language}.json")
            print(f"\n[!] {len(error_logs)} errors occurred. Saving error details to: {error_file_path}")
            with open(error_file_path, 'w', encoding='utf-8') as f:
                json.dump(error_logs, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()