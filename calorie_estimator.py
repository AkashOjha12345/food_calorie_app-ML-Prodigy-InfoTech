# Example calories (you can expand this)
food_calories = {
    "pizza": 266,
    "burger": 295,
    "salad": 152,
    "apple_pie": 237,
    "fried_rice": 163
}

def get_calories(food_name):
    return food_calories.get(food_name.lower(), "Not available")