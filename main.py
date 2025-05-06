from lib.decision_tree import DecisionTree


def main():
    dt = DecisionTree()
    dt.load_data("data/car.data")

    print("Unique values count:")
    for attr, count in dt.unique_attributes_values.items():
        print(f"Attribute {attr + 1}: {count} unique values")

    print("\nValue occurrences:")
    for attr, occurrences in dt.value_occurrences.items():
        print(f"Attribute {attr + 1}:")
        for value, count in occurrences.items():
            print(f"  Value {value}: {count} occurrences")

    entropy = dt.calculate_entropy()
    print(f"\nEntropy of the entire data set: {entropy:.4f}")

    for attr_index in dt.unique_attributes_values.keys():
        print(f"\n--- Attribute {attr_index + 1} ---")

        conditional_entropy = dt.calculate_conditional_entropy(attr_index)
        print(f"Conditional entropy: {conditional_entropy:.4f}")

        information_gain = dt.calculate_information_gain(attr_index)
        print(f"Information gain: {information_gain:.4f}")

        split_info = dt.calculate_split_information(attr_index)
        print(f"Split information: {split_info:.4f}")

        gain_ratio = dt.calculate_gain_ratio(attr_index)
        print(f"Gain ratio: {gain_ratio:.4f}")

    best_attr = max(
        dt.unique_attributes_values.keys(), key=lambda x: dt.calculate_gain_ratio(x)
    )
    print(f"\nAttribute with highest gain ratio: Attribute {best_attr + 1}")
    dt.save_tree_to_file(filename="output/decision_tree.txt")
    dt.export_tree_to_graphviz(
        dot_filename="output/decision_tree.dot", svg_filename="output/decision_tree.svg"
    )


if __name__ == "__main__":
    main()
