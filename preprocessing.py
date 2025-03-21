from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np

import xml.etree.ElementTree as ET
import re


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    def read_and_prepare_xml(path: str) -> ET.Element:
        with open(path, "r", encoding="utf-8") as file:
            xml_content = file.read().replace("&", "&amp;")
        return ET.fromstring(xml_content)
    
    def parse_alignment_str(align_str: str) -> List[Tuple[int, int]]:
        if not align_str or not align_str.strip():
            return []
        
        return [tuple(map(int, match))
                for token in align_str.split()
                for match in re.findall(r"(\d+)-(\d+)", token)]
    
    root = read_and_prepare_xml(filename)
    
    sentence_pairs = []
    alignments = []
    for s_elem in root.findall("s"):
        
        english_tokens = s_elem.find("english").text.split()
        czech_tokens = s_elem.find("czech").text.split()
        sentence_pairs.append(SentencePair(english_tokens, czech_tokens))
        
        sure_align = parse_alignment_str(s_elem.find("sure").text)
        possible_align = parse_alignment_str(s_elem.find("possible").text)
        alignments.append(LabeledAlignment(sure_align, possible_align))
        
    return sentence_pairs, alignments
    


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    src_counter = Counter()
    tgt_counter = Counter()
    
    for pair in sentence_pairs:
        src_counter.update(pair.source)
        tgt_counter.update(pair.target)
    
    if freq_cutoff is not None:
        src_tokens = [token for token, _ in src_counter.most_common(freq_cutoff)]
        tgt_tokens = [token for token, _ in tgt_counter.most_common(freq_cutoff)]
    else:
        src_tokens = list(src_counter.keys())
        tgt_tokens = list(tgt_counter.keys())
        
    source_dict = {token: idx for idx, token in enumerate(sorted(src_tokens))}
    target_dict = {token: idx for idx, token in enumerate(sorted(tgt_tokens))}
    
    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    
    for pair in sentence_pairs:
        src_indices = [source_dict[token] for token in pair.source if token in source_dict]
        tgt_indices = [target_dict[token] for token in pair.target if token in target_dict]
        
        if src_indices and tgt_indices:
            tokenized_sentence_pairs.append(
                TokenizedSentencePair(
                    source_tokens=np.array(src_indices, dtype=np.int32),
                    target_tokens=np.array(tgt_indices, dtype=np.int32)
                )
            )
    return tokenized_sentence_pairs
