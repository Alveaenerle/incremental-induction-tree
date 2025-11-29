def is_pure(data):
    classes = set(row[-1] for row in data)
    return len(classes) == 1


def majority_class(data):
    class_counts = {}
    for row in data:
        label = row[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    return max(class_counts, key=class_counts.get)
