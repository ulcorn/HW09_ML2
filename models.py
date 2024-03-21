from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple
from collections import defaultdict


import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner:
    def __init__(self, num_source_words, num_target_words, num_iters=5):
        """
        Args:
            num_source_words: размер словаря (source)
            num_target_words: размер словаря (target)
            num_iters: число итераций EM
        """
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters
        self.temp_counts = np.zeros((num_source_words, num_target_words), dtype=np.float64)
        self.row_sums = np.zeros(num_source_words, dtype=np.float64)  
        
        
    def _e_step(self, parallel_corpus: List["TokenizedSentencePair"]) -> List[np.ndarray]:
        
        posteriors = []
        for pair in parallel_corpus:
            src = pair.source_tokens
            tgt = pair.target_tokens
            
            subm = self.translation_probs[np.ix_(src, tgt)]
            
            col_sums = subm.sum(axis=0, keepdims=True)
            col_sums[col_sums < 1e-12] = 1e-12
            subm /= col_sums
            
            posteriors.append(subm.astype(np.float16))
        return posteriors

    def _compute_elbo(self,
                      parallel_corpus: List["TokenizedSentencePair"],
                      posteriors: List[np.ndarray]) -> float:
        
        elbo_val = 0.0
        for pair, posterior16 in zip(parallel_corpus, posteriors):
            src = pair.source_tokens
            tgt = pair.target_tokens
            posterior = posterior16.astype(np.float32)
            
            subm = self.translation_probs[np.ix_(src, tgt)]
            
            
            
            n = float(len(src)) if len(src) > 0 else 1.0
            
            log_prior = np.log(1.0 / n)
            with np.errstate(divide='ignore', invalid='ignore'):
                
                log_subm = np.log(subm)
                log_post = np.log(posterior)
                
                contrib = posterior * (log_prior + log_subm - log_post)
                elbo_val += contrib.sum()
                
        return float(elbo_val)

    def _m_step(self,
                parallel_corpus: List["TokenizedSentencePair"],
                posteriors: List[np.ndarray]) -> float:
        
        self.temp_counts.fill(0.0)
        self.row_sums.fill(0.0)
        
        
        for pair, posterior16 in zip(parallel_corpus, posteriors):
            src = pair.source_tokens
            tgt = pair.target_tokens
            posterior = posterior16.astype(np.float32)
            row_sum_post = posterior.sum(axis=1)
            
            for j_idx, s_word in enumerate(src):
                self.row_sums[s_word] += row_sum_post[j_idx]
                
                np.add.at(self.temp_counts[s_word], tgt, posterior[j_idx])
                
        self.translation_probs[:] = 0.0
        valid_mask = (self.row_sums > 1e-12)
        self.translation_probs[valid_mask] = (self.temp_counts[valid_mask] /
                                              self.row_sums[valid_mask, None])
        
        zero_mask = np.logical_not(valid_mask)
        if np.any(zero_mask):
            self.translation_probs[zero_mask] = 1.0 / self.num_target_words
            
        return self._compute_elbo(parallel_corpus, posteriors)

    def fit(self, parallel_corpus: List["TokenizedSentencePair"]) -> list:
        history = []
        for _ in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences: List["TokenizedSentencePair"]) -> List[List[Tuple[int,int]]]:
        
        all_alignments = []
        for pair in sentences:
            src = pair.source_tokens
            tgt = pair.target_tokens
            subm = self.translation_probs[np.ix_(src, tgt)]
            
            best_j = np.argmax(subm, axis=0)
            alignment = [(j_idx + 1, k_idx + 1) for k_idx, j_idx in enumerate(best_j)]
            all_alignments.append(alignment)
        return all_alignments


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        pass

    def _e_step(self, parallel_corpus):
        pass

    def _compute_elbo(self, parallel_corpus, posteriors):
        pass

    def _m_step(self, parallel_corpus, posteriors):
        pass


