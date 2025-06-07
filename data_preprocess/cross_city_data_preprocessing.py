import re
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import tqdm
import argparse
import os

def generate_pseudo_session(df):
    df = df.sort_values(by=['userid', 'cross_city_mode', 'datetime']).reset_index(drop=True)
    df['pseudo_session_trajectory_id'] = (
        (df['userid'] != df['userid'].shift()) | 
        (df['cross_city_mode'] != df['cross_city_mode'].shift())
    ).cumsum()
    return df


def id_encode(fit_df, encode_df, column, padding=0):
    id_le = LabelEncoder()
    id_le = id_le.fit(fit_df[column].values.tolist())
    if padding == 0:
        padding_id = padding
        encode_df[column] = [
            id_le.transform([i])[0] + 1 if i in id_le.classes_ else padding_id
            for i in encode_df[column].values.tolist()
        ]
    else:
        padding_id = len(id_le.classes_)
        encode_df[column] = [
            id_le.transform([i])[0] if i in id_le.classes_ else padding_id
            for i in encode_df[column].values.tolist()
        ]
    return id_le, padding_id

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Gowalla') 
parser.add_argument('--city_A', type=str, default='LosAngeles') 
parser.add_argument('--city_B', type=str, default='SanFrancisco') 

args = parser.parse_args()

dataset_path = f'../dataset/{args.dataset}'
save_path = f'../dataset/{args.dataset}/{args.city_A}_{args.city_B}'
os.makedirs(save_path, exist_ok=True)

if args.dataset == "Gowalla":
    if args.city_A in ['Dallas', 'Austin']:
        path_file = f'{dataset_path}/Gowalla_Dallas_Austin.csv'
    else:
        path_file = f'{dataset_path}/Gowalla_LosAngeles_SanFrancisco.csv'
else:
    raise ValueError(f'Wrong dataset name: {args.dataset} ')
data = pd.read_csv(path_file)

data.columns = ["userid", "placeid", "datetime", "lng", "lat", "spot_categ", "cross_city_mode"]


if args.dataset == "Foursquare":
    data['datetime'] = pd.to_datetime(data['datetime'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
if args.dataset == "Gowalla":
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%dT%H:%M:%SZ')
data['datetime'] = data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')

data['cat_id'] = data['spot_categ'].apply(lambda x: re.search(r'/categories/(\d+)', x).group(1) if isinstance(x, str) and re.search(r'/categories/(\d+)', x) else None)
data['cat_name'] = data['spot_categ'].apply(lambda x: re.search(r"\[\{\'url\': \'/categories/\d+\', \'name\': [\"'](.*?)[\"']\}\]", x).group(1) if isinstance(x, str) and re.search(r"\[\{\'url\': \'/categories/\d+\', \'name\': [\"'](.*?)[\"']\}\]", x) else None)

data['home_city'] = data['cross_city_mode'].str.split('_').str[0]
data['current_city'] = data['cross_city_mode'].str.split('_').str[1]

user_checkin_counts = data['userid'].value_counts()
valid_users = user_checkin_counts[user_checkin_counts >= 25].index
data = data[data['userid'].isin(valid_users)]

poi_checkin_counts = data['placeid'].value_counts()
valid_pois = poi_checkin_counts[poi_checkin_counts >= 10].index
data = data[data['placeid'].isin(valid_pois)]

home_city_records = data[data['cross_city_mode'] == f'{args.city_A}_{args.city_A}']
transition_records = data[data['cross_city_mode'] == f'{args.city_A}_{args.city_B}']
travel_city_records = data[data['cross_city_mode'] == f'{args.city_B}_{args.city_B}']

home_users = set(home_city_records['userid'].unique())
cross_city_user_records = data[data['userid'].isin(home_users)]
subdata = pd.concat([cross_city_user_records, travel_city_records])
subdata['userid'], _ = pd.factorize(subdata['userid'])
subdata['placeid'], _ = pd.factorize(subdata['placeid'])

data_filtered = subdata.copy()
data_filtered = (
    data_filtered.sort_values(by=['userid', 'cross_city_mode', 'datetime'])
    .reset_index(drop=True)
)
data_filtered = generate_pseudo_session(data_filtered)

def split_train_test_by_mode(df, train_ratio=0.8):
    train_data = []
    test_data = []
    skipped_count = 0  
    count = 0

    for traj, group in df.groupby('pseudo_session_trajectory_id'):
        group = group.sort_values(by='datetime')
        transition_group = group[group['cross_city_mode'] == f'{args.city_A}_{args.city_B}']
        travel_group = group[group['cross_city_mode'].isin([f'{args.city_A}_{args.city_A}', f'{args.city_B}_{args.city_B}'])]

        if not transition_group.empty:
            if len(transition_group) < 10:
                skipped_count += 1
                continue
            count += 1
            split_idx = int(len(transition_group) * train_ratio)
            train_data.append(transition_group.iloc[:split_idx])
            test_data.append(transition_group.iloc[split_idx:])

        if not travel_group.empty:
            count += 1
            train_data.append(travel_group)

    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    train_df = (
        train_df.sort_values(by=['pseudo_session_trajectory_id', 'datetime'])
        .reset_index(drop=True)
    )

    test_df = (
        test_df.sort_values(by=['pseudo_session_trajectory_id', 'datetime'])
        .reset_index(drop=True)
    )

    print(f"Skipped {skipped_count} trajectories with fewer than 10 records. Passed {count} trajectories.")
    return train_df, test_df


train_data, test_data = split_train_test_by_mode(data_filtered)

def split_long_trajectories(df, max_length=10):
    new_data = []
    global_counter = 0
    traj_ids = df['pseudo_session_trajectory_id'].unique()
    
    for traj_id in traj_ids:
        group = df[df['pseudo_session_trajectory_id'] == traj_id].sort_values(by=['datetime'])
        if len(group) <= max_length:
            group = group.copy()
            group['pseudo_session_trajectory_id'] = global_counter
            global_counter += 1
            new_data.append(group)
        else:
            sub_group = None 
            for i in range(0, len(group), max_length):
                temp_group = group.iloc[i:i + max_length].copy()
                if len(temp_group) <= 1:
                    temp_group['pseudo_session_trajectory_id'] = global_counter-1
                    sub_group = pd.concat([sub_group, temp_group])
                else:
                    if sub_group is not None:
                        new_data.append(sub_group)
                    sub_group = temp_group.copy()
                    sub_group['pseudo_session_trajectory_id'] = global_counter
                    global_counter += 1
            if sub_group is not None:
                new_data.append(sub_group)
    
    result = pd.concat(new_data).reset_index(drop=True)
    result = result.sort_values(by=['userid', 'pseudo_session_trajectory_id']).reset_index(drop=True)
    return result

train_data = split_long_trajectories(train_data)
test_traj_encoder, _ = id_encode(train_data, train_data, 'pseudo_session_trajectory_id')
test_traj_encoder, _ = id_encode(test_data, test_data, 'pseudo_session_trajectory_id')

train_data.to_csv(f"{save_path}/train_data.csv", index=False)
test_data.to_csv(f"{save_path}/test_data.csv", index=False)

print("Data processing complete. Train and test data saved.")
