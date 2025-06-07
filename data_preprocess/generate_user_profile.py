import json
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from tqdm.contrib.concurrent import thread_map
from cross_gpt import GPT
import re


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default="dataset")
    parser.add_argument("--dataset", type=str, default="Gowalla", choices=["Foursquare", "Gowalla"])
    parser.add_argument("--city_A", type=str, default="SanFrancisco", choices=["Washington", "Dallas", "LosAngeles"])
    parser.add_argument("--city_B", type=str, default="LosAngeles", choices=["Baltimore", "Austin", "SanFrancisco"])
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def simplify_poi_category(text):
    try:
        match = re.search(r"'name': ?'([^']+)'", text)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"Error processing POI category: {e}")
    return "Unknown"


def create_checkin_prompt(row):
    try:
        prompt = (
            "At {datetime}, user {userid} visited POI id {placeid}, "
            "(POI category: {spot_categ}, (Latitude, Longitude) is ({lat}, {lng})."
        )
        return prompt.format(**row)
    except KeyError as e:
        print(f"Error creating prompt: Missing key {e}")
        return ""


def generate_profile(user_records, llm, output_dir, num_workers):
    output_dir.mkdir(parents=True, exist_ok=True)

    def process_user_profile(user_id):
        user_df = user_records[user_records["userid"] == user_id]
        prompt_list = list(user_df["prompt"].values)

        if not prompt_list:
            print(f"No data available for user {user_id} to generate {profile_type} profile.")
            return

        output_path = output_dir / f"user_{profile_type}_profile_{user_id}.json"
        if output_path.exists():
            print(f"User {user_id} {profile_type}-term profile already exists.")
            return

        try:
            # Limit prompts to last 1000 entries for efficiency
            prompt = " ".join(prompt_list[-1000:])
            result = llm.generate_user_longterm_profile(user_id=user_id, user_history_prompt=prompt)
            else:
                raise ValueError("Invalid profile type")

            with open(output_path, "w") as f:
                json.dump(result, f, indent=4)
        except Exception as e:
            print(f"Error generating {profile_type}-term profile for user {user_id}: {e}")

    user_ids = user_records["userid"].unique()
    thread_map(process_user_profile, user_ids, max_workers=num_workers)


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    dataset_path = input_path / args.dataset / f"{args.city_A}_{args.city_B}"

    try:
        # Load and process dataset
        train_df = pd.read_csv(dataset_path / "train_data.csv")
        train_df["spot_categ"] = train_df["spot_categ"].apply(simplify_poi_category)
        train_df["prompt"] = train_df.apply(create_checkin_prompt, axis=1)

        # Separate data based on city modes
        home_records = train_df[train_df["cross_city_mode"] == f"{args.city_A}_{args.city_A}"]
        trans_records = train_df[train_df["cross_city_mode"] == f"{args.city_A}_{args.city_B}"]
        current_records = train_df[train_df["cross_city_mode"] == f"{args.city_B}_{args.city_B}"]

        profile_data = pd.concat([home_records, trans_records])

        # Initialize GPT model
        llm = GPT()

        generate_profile(
            profile_data,
            llm=llm,
            output_dir=dataset_path / "user_profiles",
            num_workers=args.num_workers,
        )
        generate_profile(
            current_records,
            llm=llm,
            output_dir=dataset_path / "user_profiles",
            num_workers=args.num_workers,
        )
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()