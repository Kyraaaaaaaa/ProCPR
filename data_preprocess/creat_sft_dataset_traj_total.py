import os
import re
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict

# Constants for prompt templates
LLAMA_PROMPT = """<s>[INST] <<SYS>>
{profile}
<</SYS>>

{instruction} [/INST] {answer} </s>"""

LLAMA_COT_PROMPT = """<s>[INST] <<SYS>>
{profile}
<</SYS>>

{instruction} [/INST]<think>{reasoning}</think><output>{answer}</output></s>"""

# Parse command-line arguments
def parse_args():
    parser = ArgumentParser(description="Process user profiles and generate training/testing data")
    parser.add_argument("--input_path", type=str, default="dataset", help="Path to the dataset folder")
    parser.add_argument("--dataset", type=str, default="Gowalla", help="Dataset name")
    parser.add_argument("--home_city", type=str, default="Austin", choices=['Dallas', 'LosAngeles'], help="Home city")
    parser.add_argument("--current_city", type=str, default="Dallas", choices=['Austin', 'SanFrancisco'], help="Current city")
    parser.add_argument("--use_cot", default=False, action="store_true", help="Enable chain-of-thought prompts")
    return parser.parse_args()

# Create long system prompt
def create_long_system_prompt(user_profile: dict, user_id: str) -> str:
    interest_preferences = ' '.join([
        f"{category}: Preference degree {preference[0]}; Time visit pattern: {preference[1]};" 
        for category, preference in user_profile.get("Interest preference", {}).items()
    ])
    mode_of_transportation = ", ".join(user_profile.get("Mode of transportation", []))
    lifestyle_characteristics = (
        user_profile.get("Lifestyle characteristics", "") 
        if isinstance(user_profile.get("Lifestyle characteristics"), str) 
        else ", ".join(user_profile.get("Lifestyle characteristics", []))
    )
    social_mode = (
        user_profile.get("Social mode", "") 
        if isinstance(user_profile.get("Social mode"), str) 
        else ", ".join(user_profile.get("Social mode", []))
    )
    movement_distance_preference = ' '.join([
        f"{distance}: Percentage of visits {percentage};" 
        for distance, percentage in user_profile.get("Movement distance preference", {}).items()
    ])

    standard_trajectories = []
    for trajectory in user_profile.get("Standard check-ins sequence", []):
        sequence = " -> ".join([f"{i}" for i in trajectory["POI movement sequence"]])
        time_pattern = trajectory["Time pattern"] 
        summary = trajectory["Mobility summary"]

        formatted_trajectory = (
            f"POI sequence: {sequence}\n"
            f"Frequent time pattern: {time_pattern}\n"
            f"Sequence mobility summary: {summary}"
        )
        standard_trajectories.append(formatted_trajectory)

    standard_trajectories_str = " ".join(
        [f"Sequence {i+1}:\n{traj}" for i, traj in enumerate(standard_trajectories)]
    )

    system_prompt = f"""
        You are user {user_id} and your preference profile is as follows:
        1. **Interest preference**: {interest_preferences}.
        2. **Mode of transportation**: {mode_of_transportation}.
        3. **Lifestyle characteristics**: {lifestyle_characteristics}.
        4. **Social mode**: {social_mode}.
        5. **Movement distance preference**: {movement_distance_preference}.
        6. **Standard check-ins sequence**: {standard_trajectories_str}
        """
    return system_prompt

# Load user profile
def get_user_profile(user_id: str, profiles_path: Path) -> Tuple[dict, dict]:
    user_profile_file = profiles_path / "user_profiles" / f"user_long_profile_{user_id}.json"

    try:
        with user_profile_file.open() as f:
            user_profile = json.load(f)
    except FileNotFoundError:
        print(f"Profile missing for user {user_id}. Using empty profile.")
        user_profile = {}
    return user_profile


# Main processing function
def main(args):
    input_path = Path(args.input_path)
    dataset_path = input_path / args.dataset 
    profiles_path = dataset_path / f"{args.home_city}_{args.current_city}" 

    with open(dataset_path / f"{args.home_city}_{args.current_city}" / "train_qa_pairs.json") as f:
        raw_data_train = json.load(f)

    with open(dataset_path / f"{args.home_city}_{args.current_city}" / "test_qa_pairs.txt") as f:
        raw_data_test = f.read().splitlines()

    raw_data_test = [{"question": q, "answer": a} for line in raw_data_test for (q, a) in [line.split("<answer>:")]]

    # raw_data_train = list(filter(lambda d: "<question>:" in d["question"], raw_data_train))
    raw_data_test = list(filter(lambda d: "<question>:" in d["question"], raw_data_test))

    def process_qa_pair(question: str, answer: str, idx: int = None) -> dict:
        answer = answer.replace("<answer>: ", "").strip()
        question = question.replace("<question>: ", "")
        current_trajectory_prompt = question.split("There are also hsitorical trajectories for similar users")[0].strip()
        user_id = re.match(r"The following data is a current trajectory of user (\d+) in current city:", current_trajectory_prompt).group(1)
        user_long_profile = get_user_profile(user_id, profiles_path)
        
        system_prompt = create_long_system_prompt(user_long_profile, user_id)
        return {"question": f"<question>: {system_prompt}{question}", "answer": f"<answer>: {answer}"}

    train_data = [process_qa_pair(**datum, idx=idx) for idx, datum in enumerate(tqdm(raw_data_train, desc="Processing train data"))]
    test_data = [process_qa_pair(**datum) for datum in tqdm(raw_data_test, desc="Processing test data")]

    with open(dataset_path / f"{args.home_city}_{args.current_city}" / "train.json", "w") as f:
        json.dump(train_data, f, indent=4)

    with open(dataset_path / f"{args.home_city}_{args.current_city}" / "test.json", "w") as f:
        json.dump(test_data, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)