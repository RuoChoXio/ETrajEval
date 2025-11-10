# chat_env.py

import json
from tqdm import tqdm
from model import BaseModel
from typing import List, Dict, Any

class ChatEnvironment:
    """
    Manages sophisticated, role-playing conversations between two BaseModel instances
    based on a structured prompt scenario, now capable of running them in parallel.
    """
    def __init__(self, model1: BaseModel, model2: BaseModel, language: str):
        """
        Initializes the chat environment.
        """
        self.model1 = model1
        self.model2 = model2
        self.language = language
        try:
            with open("prompt/sys_prompt.json", 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            self.system_prompt_template_subject = prompts['character_prompt_template_subject'][self.language]
            self.system_prompt_template_helper = prompts['character_prompt_template_helper'][self.language]
            self.aggravating_event_template = prompts['aggravating_event_template'][self.language]
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(f"FATAL ERROR: Prompt template file 'prompt/sys_prompt.json' not found or is missing required keys. Details: {e}")

    def _build_history_for_model(
        self,
        current_model_character_name: str,
        subject_character_name: str,
        character_settings: Dict[str, str],
        conversation_log: List[Dict[str, str]],
        aggravating_event_text: str = None
    ) -> List[Dict[str, str]]:
        """
        Builds the full prompt history for the current model, selecting the correct system prompt.
        """
        is_subject = (current_model_character_name == subject_character_name)
        
        if is_subject:
            template = self.system_prompt_template_subject
        else:
            template = self.system_prompt_template_helper
            
        system_content = template.format(
            character_setting=character_settings[current_model_character_name][self.language]
        )
        
        history_for_api = [{"role": "system", "content": system_content}]

        for message in conversation_log:
            role = "assistant" if message["role"] == current_model_character_name else "user"
            history_for_api.append({"role": role, "content": message["content"]})
        
        if aggravating_event_text and len(history_for_api) > 1 and history_for_api[-1]['role'] == 'user':
            event_notice = self.aggravating_event_template.format(event_text=aggravating_event_text)
            history_for_api[-1]['content'] += f" {event_notice}"

        return history_for_api

    def run_parallel_chats(self, prompt_scenarios: List[Dict[str, Any]], max_turns: int) -> List[Dict[str, Any]]:
        """
        Runs a BATCH of conversations in a parallel manner.
        """
        states = {}
        for scenario in prompt_scenarios:
            name1, name2 = list(scenario['character_settings'].keys())
            initial_dialogue = [{"role": name, "content": message[self.language]} for name, message in scenario['initial_dialogue'].items()]
            states[scenario['id']] = {
                "conversation_log": initial_dialogue,
                "settings": scenario['character_settings'],
                "aggravating_events": {event['turn']: event[self.language] for event in scenario.get('aggravating_events', [])},
                "model_map": {name1: self.model1, name2: self.model2},
                "current_speaker_name": name2 if len(initial_dialogue) % 2 != 0 else name1,
                "subject_character_name": name1,
                "prompt_scenario": scenario
            }

        active_conversations = dict(states)
        
        with tqdm(total=max_turns, desc=f"  - Processing Batch (size: {len(prompt_scenarios)})", leave=False) as pbar:
            for turn_num in range(max_turns * 2):
                if not active_conversations:
                    break
                
                histories_for_model1, ids_for_model1 = [], []
                histories_for_model2, ids_for_model2 = [], []

                for convo_id, state in active_conversations.items():
                    current_model = state["model_map"][state["current_speaker_name"]]
                    
                    turn_index = (len(state["conversation_log"]) - len(state['prompt_scenario']['initial_dialogue'])) // 2 + 1
                    event_text_for_turn = None
                    if state["current_speaker_name"] == state["subject_character_name"]:
                         if turn_index in state["aggravating_events"]:
                            event_text_for_turn = state["aggravating_events"][turn_index]
                    
                    history = self._build_history_for_model(
                        state["current_speaker_name"],
                        state["subject_character_name"],
                        state["settings"],
                        state["conversation_log"],
                        event_text_for_turn
                    )
                    
                    if current_model == self.model1:
                        histories_for_model1.append(history)
                        ids_for_model1.append(convo_id)
                    else:
                        histories_for_model2.append(history)
                        ids_for_model2.append(convo_id)

                if histories_for_model1:
                    responses = self.model1.batch_generate(histories=histories_for_model1)
                    for i, convo_id in enumerate(ids_for_model1):
                        state = active_conversations[convo_id]
                        state["conversation_log"].append({"role": state["current_speaker_name"], "content": responses[i]})
                        name1, name2 = list(state["settings"].keys())
                        state["current_speaker_name"] = name2 if state["current_speaker_name"] == name1 else name1

                if histories_for_model2:
                    responses = self.model2.batch_generate(histories=histories_for_model2)
                    for i, convo_id in enumerate(ids_for_model2):
                        state = active_conversations[convo_id]
                        state["conversation_log"].append({"role": state["current_speaker_name"], "content": responses[i]})
                        name1, name2 = list(state["settings"].keys())
                        state["current_speaker_name"] = name2 if state["current_speaker_name"] == name1 else name1
                
                if turn_num % 2 == 1:
                    pbar.update(1)
        
        final_conversations = []
        for convo_id, state in states.items():
            final_conversations.append({
                "id": convo_id,
                "psychological_theory": state["prompt_scenario"]["psychological_theory"],
                "conversations": state["conversation_log"]
            })
            
        return final_conversations