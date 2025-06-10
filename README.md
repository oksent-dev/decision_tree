# Decision Tree Analysis Tool

This project provides a command-line tool for analyzing datasets using decision tree metrics. It is implemented in Python and is designed to help users understand the structure and information content of their data, particularly for classification tasks.

## Features

- Loads data from a text file (see `data/` directory for examples)
- Calculates and displays:
  - Unique values count for each attribute
  - Occurrences of each value per attribute
  - Entropy of the entire dataset
  - Conditional entropy, information gain, split information, and gain ratio for each attribute
  - Identifies the attribute with the highest gain ratio
- Exports the resulting decision tree to a text file and Graphviz DOT/SVG formats

4. Results will be printed to the console and output files will be saved in the `output/` directory.

## Data Format

- Each line in the data file should represent a data instance, with attribute values separated by spaces or commas.
- Example (`gielda.txt`):
  ```
  1,0,1,1
  0,1,0,1
  ...
  ```

## Output

- Console output includes statistics and information-theoretic measures for each attribute.
- Files generated in `output/`:
  - `decision_tree.txt`: Textual representation of the decision tree
  - `decision_tree.dot`: Graphviz DOT file for visualization
  - `decision_tree.svg`: SVG image of the tree (requires Graphviz)

## Requirements

- Python 3.10 or newer
- [Graphviz](https://graphviz.gitlab.io/download/) (for SVG export)

## License

This project is for educational and research purposes.
