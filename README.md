# Cross-lingual Transfer Using Stacked Language Adapters
Repository containing the code for the Research Masters thesis project, Cross-lingual Transfer Using Stacked Language Adapters.

This work was carried out by Marcell Fekete as part of an internship project with TAUS.

## Project Goal

The main goal of the project is to investigate cross-lingual transfer between stacked language adapters using a cross-lingual sentence retrieval task.

*****

## Getting started

To start off, create a new virtual environment using Python 3.6+, and run the following line.

`pip install -r requirements.txt`

This will install most of the required packages to run the scripts in this repository. The only exception is `faiss`, the library used for similarity search between sentence embeddings. Installing this library might cause problems depending on the operating system.

I recommend running experiments using Google Colab, in which case installing `faiss` on a local machine is not necessary. Otherwise, use the following line to install the library with GPU access:

`pip install faiss-gpu`

And run the following line for installations that will use only a CPU:

`pip install faiss-cpu`

*****

## Experiments

### On local machine

Run all scripts from the main folder. To run experiments on a local machine, run `src/main.py`. To provide configuration to the experiments, edit `config.ini`.

Change the folder for the relevant language pair in the lines `path_to_source`, `path_to_target`, and `path_to_labels`, as well as `output_directory` and `save_source_path`.
Experimental setups can be provided by adding experiment names in the `experiments` line, divided by whitespaces. More experiments can be carried out with the same config file if the option on line `multiconfig` is set to True.

In lines `source_adapters` and `target_adapters`, provide language adapter IDs from AdapterHub. Use whitespaces to divide adapters that are meant to be used in the same experiment. If adapters are meant to be stacked, divide the adapter IDs using a comma and no whitespace. If instead of adapter ID, the string "None" is used, no adapter is added.

Examples:

```
source_adapters = None en en
target_adapters = None hu hu,fi
```

In this example, no language adapter is loaded to process source or target sentences in the first experiment (`None` and `None`). In the second experiment, source sentences are processed using the English language adapter (`en`), and target sentences are processed using the Hungarian one (`hu`). Finally, in the third experiment, the English adapter is still used to process the source sentences (`en`), but the Finnish language adapter is stacked on top of the Hungarian one to process the Hungarian sentences (`hu,fi`).

### On Google Colab

Start with uploading the `colab` folder to Google Drive in a separate folder named `Marcell_Fekete_Cross-Lingual_Transfer_Using_Stacked_Language_Adapters`. The full filepath to the folder should be the following:

```
/contents/gdrive/MyDrive/Marcell_Fekete_Cross-Lingual_Transfer_Using_Stacked_Language_Adapters/colab
```

Either open the `colab/Tatoeba_experiment.ipynb` notebook as a Google Colab notebook or copy-paste its content into a new Google Colab notebook. Run all cells. Experiments are started when the following cell is run:

```
!python gdrive/MyDrive/Marcell_Fekete_Cross-Lingual_Transfer_Using_Stacked_Language_Adapters/src/main.py
```

Before running this, modify the uploaded `colab/config.ini` according to the outlines described in the Section 'On local machine'.

*******

## Structure

### `colab` folder

Upload it to Google Drive in order to use Google Colab to carry out experiments.

### `data` folder

Contains evaluation data for the cross-lingual sentence retrieval task. Each folder contains a `source.txt` with English sentences, a `shuffled.txt` with target sentences, and a `labels.txt` with sentence ID for the correct translations.

Language names are abbreviated using ISO codes. A full list is on https://iso639-3.sil.org/code_tables/639/data.

* amh: Amharic
* chm: Meadow Mari
* eng: English
* fra: French
* hun: Hungarian
* ind: Indonesian
* isl: Icelandic
* sme: Northern Sami
* vie: Vietnamese

### `notebooks` folder

Contains Jupyter notebooks that were used to prepare and analyse experiments.

`morphological_complexity.ipynb` contains code to calculate overall type-token ratio (TTR) and moving average type-token ratio scores (MATTR).

`Plot_percentages.ipynb` and `Generate_example_graphs.ipynb` are used to create various graphs for analysis and visualisation.

`Analyse correlation between adapter training data difference and performance.ipynb` does what it says.

`Error analysis.ipynb` is used to carry out qualitative error analysis.

### `resources` folder

`parallel_bible_corpus` is where the full Parallel Bible Corpus is expected to be. It is not uploaded due to its large size, but can be obtained from the following link:

http://sjmielke.com/data/raw-mpost-bibles.tar.gz

`saved_embeddings` may contain saved sentence embeddings of the English evaluation sets to speed up experiments (since then they can be loaded from the files)

### `results` folder

`outputs` contains the scores per experiment per language pair that were outputted by the experimental code

`tables` contains aggregated codes for each language as well as per individual language

`figures` contains different graphs created to represent experimental results and to use for quantitative error analysis

### `src` folder

The `src` folder contains most of the code that is essential for the thesis, including experiment utilities (`utils`), and the `main.py` script for running experiments.

It also contains `translation.py` that allows checking the quality of the evaluation data.

`create_tables.py` and `create_graphs.py` aggregates experimental results and outputs them.

Finally, `extract_factors.py` allows qualitative error analysis following principles of [Explainaboard](https://aclanthology.org/2021.acl-demo.34/).

### Thesis report

The full thesis report for the project Cross-lingual Transfer Using Stacked Language Adapters is uploaded to the repository.

### `Thesis_sheets.xlsx`

This Excel sheet contains various information I had about the individual languages and benchmarks. I reference it in Appendix B of my thesis, and it is one of the most essential tools that I used for my work.

******

## Author:
* Marcell Fekete (student number: 2695076)


