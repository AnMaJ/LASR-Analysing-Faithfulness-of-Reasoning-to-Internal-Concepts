# Feature Extraction

This module extracts features from attribution graphs for training the verification model. Use the command below to extract features from your attribution graphs.

```bash
python feature_extraction.py \
    --expressions_json /path/to/expression/json/file \
    --graph_name_prefix graph_prefix \
    --feature_type "advanced_graph_features" \
    --before_after before_after \
    --num_workers 64 \
    --batch_size 32 \
    --use_cpu \
    --graph_dir /path/to/your/graph/files/ \
    --save_data /path/to/your/.npz/file \
    --save_format "numpy"

```
- `--expressions_json`: Path to the file containing the expressions for attribution analysis.
- `--graph_name_prefix`: Prefix added to the graph file for identification purposes.
- `--graph_dir`: Directory where all attribution graphs are stored.
- `--feature_type`: Multiple feature extraction types are supported. The main work uses `advanced_graph_features`.
- `--before_after`: Controls whether to use the token position before or after the step when computing the attribution graph. Accepts `before` or `after`. See **Appendix B.2** of the paper for details.
- `--save_format`: Output format for the data, either `numpy` or `pickle`.
- `--save_data`: Destination path for saving the extracted data.

Additionally, this script can generate feature visualizations (e.g., Figure 3 in the paper) from the computed data.

```bash
python feature_extraction.py \
    --load_data /path/to/your/.npz/file \
    --load_format "numpy" \
    --method "pca" \
    --save_plot /path/to/your/plot/file
```
- `--load_data`: Path to the data file you want to visualize.
- `--load_format`: Format of the data, either `numpy` or `pickle`.
- `--method`: Dimensionality reduction method for plotting, either `pca` or `tsne`.
- `--save_plot`: Destination path for saving the plot.
