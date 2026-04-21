def get_next_difficulty(accuracy, avg_time, current):
    if accuracy >= 0.8 and avg_time < 15:
        if current == "easy":
            return "medium"
        if current == "medium":
            return "hard"
    elif accuracy < 0.5:
        if current == "hard":
            return "medium"
        if current == "medium":
            return "easy"
    return current