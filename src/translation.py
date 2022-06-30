"""
Functions that allow testing the quality of Tatoeba test data.
@author: Marcell Fekete
"""


import os
import pandas as pd
from googletrans import Translator


generic_path = "results/outputs/"


def translate_sentence(sentence, trg, src, sequence=False):
    """
    Given a sentence, it returns a translation.

    :param sentence: input sentence or sentences
    :param src: source language code, if None, then autodetect
    :param trg: target language code
    :param sequence: set to True if input is a sequence
    :return: translated sentence
    """
    translator = Translator()

    if src:
        translation = translator.translate(sentence, src=src, dest=trg)
    else:
        translation = translator.translate(sentence, dest=trg)

    if sequence is True:
        return [trans.text for trans in translation]
    else:
        return translation.text


def sample_and_translate(language_pair, sample_size=50, src="en"):
    """
    Sample sentences from input,
    and translate and present them.

    :param language_pair: eng and target language pair
    :param sample_size: the number of sentences to randomly sample from the input
    :param src: source language
    :return: None
    """
    input_path = os.path.join(generic_path, language_pair, "no-lang/output.tsv")
    df = pd.read_csv(input_path, sep="\t", index_col=0).sample(n=sample_size)
    english_sentences = df["Gold"].tolist()
    source_sentences = df["Sentence"].tolist()

    translations = translate_sentence(source_sentences, trg="en",
                                      src=src, sequence=True)

    src_lg, trg_lg = language_pair.split("-")

    output_df = pd.DataFrame.from_dict({
        src_lg: english_sentences,
        trg_lg: translations
    })

    output_df.to_csv(os.path.join("check_translations", f"{language_pair}.tsv"), sep="\t")
    print(output_df)
