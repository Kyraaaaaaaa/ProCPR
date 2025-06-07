import openai
import json
import tiktoken


class GPT:

    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        self.model = model
        # 设置 OpenAI API 密钥和基础 URL
        openai.api_key = ""
        openai.base_url = ""
        openai.default_headers = {"x-foo": "true"}
        
    def generate_poi_intention(self, system_prompt: str, inputs: str, targets: str, **kwargs) -> dict:
        prompt = '{system_prompt}\n\n{trajectory}\n\n{targets}\n\nWith a step-by-step reasoning, think and suggest why they might have intended on visiting that POI. Analyze the user\'s profiles, previous visits in trajectory, time of visit of the POI, and their routines. DO NOT mention or analyze the selected POI id at all. You may suggest POI categories that they might have wanted to visit at that hour and based on their profile and trajectory. Return your response in JSON format: {{"profile_analysis": ..., "trajectory_analysis": ..., "time_of_visit_analysis": ..., "routines_and_preferences_analysis": ..., "potential_categories_of_interest_analysis": ..., "verdict": ...}}. All values should be in string format.'

        response = openai.ChatCompletion.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(system_prompt=system_prompt, trajectory=inputs, targets=targets),
                }
            ],
        )

        result = response.choices[0].message.content
        return json.loads(result)
    

    def generate_user_longterm_profile(self, user_id: str, user_history_prompt: str) -> dict:
        prompt = (
            f"Given the check-ins in the current of user {user_id}: {user_history_prompt}, summarize a 200-word structured short-term behavior profile summary without specific POI IDs:"
            "Return the output strictly in JSON format as described below: "
            "{{"
            '"Interest preference": '
            '{{'
            '"POI category name": [degree of interest (1–5), "time pattern (e.g., morning/evening, highlight weekday/weekends, stay duration estimate)"], ...'
            '}}, '
            '"Mode of transportation": [walking, driving, public transport, ...], '
            '"Lifestyle characteristics": ["routine-oriented", "exploration-oriented", or "mixed"], '
            '"Social mode": ["solo", "group", or "mixed"], '
            '"Movement distance preference": '
            '{{'
            '"Short-distance (0–5 km)": "Percentage of visits (e.g., 70%)", '
            '"Medium-distance (5–10 km)": "Percentage of visits", '
            '"Long-distance (10+ km)": "Percentage of visits"'
            '}}, '
            '"Standard check-ins sequence": '
            '['
            '{{'
            '"POI movement sequence": ["POI category name 1", "POI category name 2", ..."], '
            '"Time pattern": "Typical time range and pattern of this sequence", '
            '"Mobility summary": "Description of movement between POIs, including trajectory continuity and temporal alignment, emphasizing patterns in timing, areas, or repetition"'
            '}}, '
            '... '
            ']'
            '}}'
        )

        
        enc = tiktoken.get_encoding("cl100k_base")

        # Function to check the token count
        def get_token_count(text):
            return len(enc.encode(text))
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                response_format={"type": "json_object"},
                temperature=0,
                top_p=1,
                messages=[
                        {
                        "role": "user",
                        "content": prompt.format(user_id=user_id, user_history_prompt=user_history_prompt),
                        }
                ],
                max_tokens=3000
            )

            result = response.choices[0].message.content
            return json.loads(result)
        except openai.OpenAIError as e:  # Use InvalidRequestError
            # Catch the error when context length exceeds
            print(f"Error: {e}")
            
            # Check the length of the user_history_prompt
            token_count = get_token_count(user_history_prompt)
            print(f"Token count of user_history_prompt: {token_count}")
            
            # You can also print part of the user_history_prompt if needed:
            print(f"First 500 characters of user_history_prompt: {user_history_prompt}")  # For debugging purposes

            # Handle the error (e.g., truncate the text, split the input, etc.)
            # Example: return None or handle differently
            return None
        except json.JSONDecodeError as e:
            print(f"User{user_id}, Error decoding JSON: {e}")


        # Function to check the token count
        def get_token_count(text):
            return len(enc.encode(text))
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                response_format={"type": "json_object"},
                temperature=0,
                top_p=1,
                messages=[
                        {
                        "role": "user",
                        "content": prompt.format(user_id=user_id, user_history_prompt=user_history_prompt),
                        }
                ],
                # max_tokens=1000
            )

            result = response.choices[0].message.content
            return json.loads(result)
        except openai.OpenAIError as e:  # Use InvalidRequestError
            # Catch the error when context length exceeds
            print(f"Error: {e}")
            
            # Check the length of the user_history_prompt
            token_count = get_token_count(user_history_prompt)
            print(f"Token count of user_history_prompt: {token_count}")
            
            # You can also print part of the user_history_prompt if needed:
            print(f"First 500 characters of user_history_prompt: {user_history_prompt}")  # For debugging purposes

            # Handle the error (e.g., truncate the text, split the input, etc.)
            # Example: return None or handle differently
            return None
        except json.JSONDecodeError as e:
            print(f"User{user_id}, Error decoding JSON: {e}")
