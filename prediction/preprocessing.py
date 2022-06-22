"""
Preprocessing tweets functions:
- Remove punctation
- Use TweetTokenizer (NLTK) to keep hashtags and twitter related content
- Lowercasing
- Remove stopwords
- Lemmatisation & Stemming
- Remove singular use words (the model that we are producing will not rely on the flow of text, therefore the context or meaning of singular terms will not help us. 
- Remove links, videos and mentions
- Process emoticons
"""

import pandas as pd
import string
import pickle
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import FreqDist
from pandas import Series



# 'Emoji_Dict.p'- download link https://drive.google.com/open?id=1G1vIkkbqPBYPKHcQ8qy0G2zkoab2Qv4v


PUNCTUATION_LIST = list(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
"""
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
"""
class DimensionalityReduction:

	def __init__(self) -> None:
		self.single_use_words = []
		pass
		

	def remove_punctuation(self, word_tokens:list) -> list:
		"""Remove punctuation from list of words

		Args:
			word_list (list): list of words

		Returns:
			list: List of words with punctuation removed
		"""
		return [w for w in word_tokens if w not in PUNCTUATION_LIST]



	def remove_stopwords(self, word_tokens:list) -> list:
		"""Remove words that do not add any information to analysis but increase the size

		Args:
			word_tokens (list): List of word tokens

		Returns:
			list: List of word token minus identified stopwords
		"""
		## STOPWORDS imported from NLTK corpus
		return [w for w in word_tokens if w not in STOPWORDS]



	def generate_single_use_word_list(self, corpus:Series) -> None:
		"""Generate list of words that only appear once. Only used in training.

		Args:
			corpus (Series): List of word tokens
		"""
		corpus_tokens = corpus.astype('str').sum()
		corpus_freq_dist = FreqDist(corpus_tokens)
		self.single_use_words = [w for w in corpus_freq_dist.most_common() if w[1] == 1]



	def remove_single_use_words(self, word_tokens:list) -> list:
		"""Removes words outlined as appearing only once.
			Only used in the training phase, not in the prediction preprocessing

		Args:
			word_tokens (list): List of word tokens

		Returns:
			list: List of word tokens minus single use words
		"""

		if len(self.single_use_words) < 1:
			print('Single use words not found')
		
		return [w for w in word_tokens if w not in self.single_use_words]



	def remove_standardised_references(self, sentence:str) -> str:
		"""Removing identified standard references that have been used throughout the text

		Args:
			sentence (str): Sentence

		Returns:
			str: amended_sentence if any standard references have been found else sentence
		"""

		sentence = re.sub(r'https?:\/\/\S+', '', sentence) # URLs
		sentence = re.sub(r'@mention', '', sentence) # @mention
		sentence = re.sub(r'@mention', '', sentence) # @mention
		sentence = re.sub(r'{link}', '', sentence) # {link}
		sentence = re.sub(r'&[a-z]+;', '', sentence) # HTML code
		sentence = re.sub(r"\[video\]", '', sentence) # [video]
		return sentence




class TextConversion:

	def __init__(self) -> None:
		self.tknzr = TweetTokenizer()
		self.lmtzr = WordNetLemmatizer()

		emoji_dict = pickle.load(open('Emoji_Dict.p', 'rb'))
		self.emoji_dict = {v: k for k, v in emoji_dict.items()}
		pass

	def convert_sentence_to_lowercase(self, sentence:str) -> str:
		"""Removes upper case letters from string for better matching rate

		Args:
			sentence (str): Sentence

		Returns:
			str: Lowercase sentence
		"""
		return sentence.lower()


	def convert_word_list_to_lowercase(self, word_tokens:list) -> list:
		"""Removes upper case letters from list of words for better matching rate

		Args:
			word_tokens (list): List of work tokens

		Returns:
			list: List of lowercase word tokens
		"""
		return [w.lower() for w in word_tokens]



	def text_lemmatisation(self, word_tokens:list) -> list:
		"""Reduce word to root e.g. running to run

		Args:
			word_tokens (list): List of word tokens

		Returns:
			list: List of word token lemmatised
		"""
		word_tokens_lemmatised = []
		tokens_pos_tagged = nltk.pos_tag(word_tokens)

		for token in tokens_pos_tagged:
			## Better performance when using part of speech tag
			word, pos_tag = token[0], self.get_wordnet_pos(token[1])
			if pos_tag:
				token_lemmatised = self.lmtzr.lemmatize(word, pos=pos_tag)
				word_tokens_lemmatised.append(token_lemmatised)
			else:
				word_tokens_lemmatised.append(word)

		return word_tokens_lemmatised

	def convert_emoticons(self, sentence:string) -> string:
		"""NOT IN USE - due to processing times
		Converts basic emoticons into text

		Args:
			sentence (string): Sentence

		Returns:
			string: Sentence with emoticons converted
		"""
		try:
			for emot in self.emoji_dict:
				sentence = re.sub(r'('+emot+')', "_".join(self.emoji_dict[emot].replace(",","").replace(":","").split()), sentence)
		except Exception as e:
			print('Unable to convert emoticons in sentence:', sentence)
		return sentence



	def tokenise_text(self, sentence:string) -> list:
		"""Converts sentence into list of words

		Args:
			sentence (string): Sentence

		Returns:
			list: List of word tokens
		"""
		return self.tknzr.tokenize(sentence)



	def get_wordnet_pos(self, treebank_tag:str) -> str:
		"""Lemmatiser takes in a restricted number of POS tags so we convert tags to their simple form

		Args:
			treebank_tag (str): POS tag

		Returns:
			str: Simplified POS form
		"""

		if treebank_tag.startswith('J'):
			return wordnet.ADJ
		elif treebank_tag.startswith('V'):
			return wordnet.VERB
		elif treebank_tag.startswith('N'):
			return wordnet.NOUN
		elif treebank_tag.startswith('R'):
			return wordnet.ADV
		else:
			return ''


