import pandas as pd
import math
import json
import pickle
def predict_mpg(config):
    try:
        ruleViolated = []
        # Convert the input to a JSON string if it's not already
        if isinstance(config, dict):
            config = json.dumps(config)

        # Convert JSON string to dictionary
        data_dict = json.loads(config)

        # Create DataFrame from dictionary
        df = pd.DataFrame(data_dict, index=[0])

        # Assuming these are the methods of your model to check rule 001 and 002
        y_pred_1 = check_rule_001(df)
        y_pred_2 = check_rule_002(df, df['entityId'][0])  # Assuming entityId is unique for each transaction
        y_pred_3 =  check3(config)
        if y_pred_1: 
            ruleViolated.append("RULE-001")
        
        elif y_pred_2 :
            ruleViolated.append("RULE-002")
        elif df['merchantCategoryCode'][0] == "9565":
            ruleViolated.append("RULE-004")
        else:
            ruleViolated.append("RULE-003")

    except Exception as e:
        return f'Error: {str(e)}'



def check_rule_001(transactions_df):
    # Selecting relevant columns for the condition
    relevant_columns = [
        'transactionAmount', 'dateTimeTransaction', 'timeLocalTransaction',
        'dateLocalTransaction', 'merchantCategoryCode', 'acquiringInstitutionCode',
        'cardAcceptorId', 'cardBalance', 'channel', 'transactionOrigin',
        'transactionType', 'entityId', 'latitude', 'longitude'
    ]
    
    # Filter dataframe to include only relevant columns
    relevant_transactions = transactions_df[relevant_columns]

    # Convert transactionAmount and cardBalance to numeric
    relevant_transactions['transactionAmount'] = pd.to_numeric(relevant_transactions['transactionAmount'], errors='coerce')
    relevant_transactions['cardBalance'] = pd.to_numeric(relevant_transactions['cardBalance'], errors='coerce')

    # Convert dateTimeTransaction to timestamp
    relevant_transactions['dateTimeTransaction'] = pd.to_datetime(relevant_transactions['dateTimeTransaction'], errors='coerce')

    # Filter transactions within the last 12 hours
    twelve_hours_ago = pd.Timestamp.now() - pd.Timedelta(hours=12)
    recent_transactions = relevant_transactions[relevant_transactions['dateTimeTransaction'] >= twelve_hours_ago]

    # Group by entityId and calculate cumulative transaction amount for each entity
    cumulative_amount_per_entity = recent_transactions.groupby('entityId')['transactionAmount'].sum()

    # Calculate the maximum card balance for each entity
    max_card_balance_per_entity = relevant_transactions.groupby('entityId')['cardBalance'].max()

    # Check if the total cumulative amount >= 70% of the card balance and balance >= Rs 3,00,000 for any entity
    for entity_id, cumulative_amount in cumulative_amount_per_entity.items():
        max_balance = max_card_balance_per_entity.get(entity_id, 0)
        if cumulative_amount >= 0.7 * max_balance and max_balance >= 300000:
            return True

    return False

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles.
    return r * c

def check_rule_002(transactions_df, entity_id):
    # Selecting relevant columns for the condition
    relevant_columns = [
        'transactionAmount', 'dateTimeTransaction', 'timeLocalTransaction',
        'dateLocalTransaction', 'merchantCategoryCode', 'acquiringInstitutionCode',
        'cardAcceptorId', 'cardBalance', 'channel', 'transactionOrigin',
        'transactionType', 'entityId', 'latitude', 'longitude'
    ]
    
    # Filter dataframe to include only transactions for the specified entityId
    entity_transactions = transactions_df[transactions_df['entityId'] == entity_id][relevant_columns]

    # Convert transactionAmount and cardBalance to numeric
    entity_transactions['transactionAmount'] = pd.to_numeric(entity_transactions['transactionAmount'], errors='coerce')
    entity_transactions['cardBalance'] = pd.to_numeric(entity_transactions['cardBalance'], errors='coerce')

    # Convert dateTimeTransaction to timestamp
    entity_transactions['dateTimeTransaction'] = pd.to_datetime(entity_transactions['dateTimeTransaction'], errors='coerce')

    # Filter transactions within the last 12 hours
    twelve_hours_ago = pd.Timestamp.now() - pd.Timedelta(hours=12)
    recent_transactions = entity_transactions[entity_transactions['dateTimeTransaction'] >= twelve_hours_ago]

    # Check if total transaction amount within 12 hours is more than Rs 1,00,000
    total_transaction_amount = recent_transactions['transactionAmount'].sum()
    if total_transaction_amount < 100000:
        return False

    # Group transactions by location and count unique locations
    unique_locations = recent_transactions.groupby(['latitude', 'longitude']).size().reset_index(name='count')

    # Check if the user transacts from more than 5 locations with a minimum difference of 200KM between two locations
    if len(unique_locations) <= 5:
        return False

    # Check the distance between each pair of locations
    for i in range(len(unique_locations)):
        for j in range(i + 1, len(unique_locations)):
            lat1, lon1 = unique_locations.iloc[i]['latitude'], unique_locations.iloc[i]['longitude']
            lat2, lon2 = unique_locations.iloc[j]['latitude'], unique_locations.iloc[j]['longitude']
            distance = calculate_distance(lat1, lon1, lat2, lon2)
            if distance < 200:
                return False

    return True

def check_rules(json_data):
    # Convert JSON data to DataFrame
    transactions_df = pd.DataFrame(json_data)

    # Check Rule 001
    rule_001_result = check_rule_001(transactions_df)

    # Check Rule 002
    if 'entityId' in json_data:
        rule_002_result = check_rule_002(transactions_df, json_data['entityId'])
    else:
        rule_002_result = False

    return {
        'rule_001': rule_001_result,
        'rule_002': rule_002_result
    }


def check3(config):
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)

    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    y_pred = model.predict(df)

    if y_pred==-1:
        return True
    else:
        return False