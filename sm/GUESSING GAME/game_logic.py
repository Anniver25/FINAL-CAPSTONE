# game_logic.py
def get_performance_rating(guesses):
    
    if guesses <= 5:
        return "Excellent"
    elif 6 <= guesses <= 10:
        return "Very Good"
    elif 11 <= guesses <= 20:
        return "Average"
    else:
        return "Needs Improvement"
