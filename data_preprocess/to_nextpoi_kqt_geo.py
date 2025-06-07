import pandas as pd
import json
import argparse
import io
import pandas as pd
import json
import sys
import math
from tqdm import tqdm

import re
def simplify_poi_category(text):
    # Modify this regular expression pattern based on the specific substitution you need
    match = re.search(r"\[\{\'url\': \'/categories/\d+\', \'name\': [\"'](.*?)[\"']\}\]", text)
    if match:
        return match.group(1)
    return text


def generate_qa_pairs(main_data, if_test=False, args=None):
    # Sort the dataframe by UserId, pseudo_session_trajectory_id, and timestamp
    main_data = main_data.sort_values(by=['userid', 'pseudo_session_trajectory_id', 'datetime'])
    # List to store the QA pairs
    qa_pairs = []

    # Iterate over each user
    print(len(main_data['userid'].unique()))
    for user in tqdm(main_data['userid'].unique()):
        user_data = main_data[main_data['userid'] == user]

        # Iterate over each unique trajectory for the user based on 'pseudo_session_trajectory_id'
        for traj_id in user_data['pseudo_session_trajectory_id'].unique():
            user_trajectory_data = user_data[user_data['pseudo_session_trajectory_id'] == traj_id]

            num_traj = user_trajectory_data.shape[0]
            if num_traj <= 1:
                continue

            if if_test:
                user_trajectory_data = user_trajectory_data.sort_values(by='datetime', ascending=True).iloc[-100:]

            user_trajectory_data.reset_index(drop=True, inplace=True)
            # Create the question based on the current trajectory (excluding the last entry) and historical data
            question_parts = [f"<question>: The following data is a current trajectory of user {user} in current city:"]
            for i, row in user_trajectory_data.iloc[:-1].iterrows():
                question_parts.append(
                    f"At {row['datetime']}, visited POI id {row['placeid']}, "
                    f"(POI category: {row['spot_categ']}, (Latitude, Longitude) is ({row['lat']}, {row['lng']}).)"
                )
            if not user_sim_data.empty:
                if len(user_trajectory_data.iloc[:-1]) > 0:
                    question_parts.append(f"There are also hsitorical trajectories for similar users of user {user}:")
                else:
                    question_parts = [f"There are hsitorical trajectories for similar users of user {user}:"]
                for _, row in user_sim_data.iterrows():
                    question_parts.append(
                        f"At {row['datetime']}, visited POI id {row['placeid']}, (POI category: {row['spot_categ']}, (Latitude, Longitude) is ({row['lat']}, {row['lng']}).)"
                        )

            # Create the final question string
            question = " ".join(question_parts)
            value = {'Dallas': 11374, 'Austin': 4721, 'LosAngeles': 3207, 'SanFrancisco': 3682}[args.current_city]
            question += f"Given the user's current trajectory and user's profile information, At {user_trajectory_data.iloc[-1]['datetime']}, Which POI id will user {user} visit? Note that POI id is an integer in the range from 0 to {value}."
            # Form the answer based on the last entry of the current trajectory
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['datetime']}, user {user} will visit POI id {user_trajectory_data.iloc[-1]['placeid']}."
            # Append the question-answer pair to the list
            qa_pairs.append((question, answer))
    return qa_pairs


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process dataset names.")

    # Add an argument for the dataset name
    parser.add_argument("--input_path", type=str, default="dataset")
    parser.add_argument("--dataset", type=str, default="Gowalla")
    parser.add_argument("--home_city", type=str, default="LosAngeles", choices=['Dallas','LosAngeles'])
    parser.add_argument("--current_city", type=str, default="SanFrancisco", choices=['Austin','SanFrancisco'])

    # Parse the arguments
    args = parser.parse_args()

    # Your processing code here
    print(f"Processing dataset: {args.dataset}_{args.home_city}_{args.current_city}")
    path = f"{args.input_path}/{args.dataset}/{args.home_city}_{args.current_city}"
    # Read the data
    train_data = pd.read_csv(f'{path}/train_data.csv')
    test_data = pd.read_csv(f'{path}/test_data.csv')
    
    train_data['spot_categ'] = train_data['spot_categ'].apply(simplify_poi_category)
    test_data['spot_categ'] = test_data['spot_categ'].apply(simplify_poi_category)
    
    # Generate the QA pairs
    qa_pairs_train = generate_qa_pairs(train_data, if_test=False, args=args)
    qa_pairs_test = generate_qa_pairs(test_data, if_test = True, args=args)

    # Save the train QA pairs in JSON format
    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    with open(f'{path}/train_qa_pairs.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)


    # Save the test QA pairs in TXT format
    with open(f'{path}/test_qa_pairs.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')


if __name__ == "__main__":
    main()

