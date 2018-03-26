import os
import pandas as pd

from multiprocessing import Pool

from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

if not os.path.isfile('../data/breitbart_data.csv'):
	articles = pd.read_csv("../data/breitbart_articles_fb.csv", dtype={"article_id": 'str'})

	articles.loc[articles['tags'].notnull() & articles['tags'].str.match('/tag/islam/|/jihad/|/tag/jihad/|/tag/muslim/|/tag/radical-islam/|/tag/muslims/|/tag/sharia-law/|/tag/sharia/|/tag/islamism/|/tag/muslim-immigration/|/tag/sunni/|/tag/ramadan/|/tag/shiite/|/tag/islamophobia/|/tag/mosque/|/tag/koran/|/tag/caliphate/|/tag/islamic-terrorism/|/tag/hijab/|/tag/quran/|/tag/muhammad/|/tag/mohammed/|/tag/islamic-extremism/|/tag/hajj/|/tag/shia/|/tag/allah/|/tag/mosques/|/tag/burqa/|/tag/niqab/|/tag/islamist/|/tag/imam/|/tag/muslim-faith/|/tag/fatwa/|/tag/islamists/|/tag/no-go-zones/|/tag/anti-islam/|/tag/halal/|/tag/islamisation/|/tag/muslim-terrorism/|/tag/british-muslims/|/tag/holy-war/|/tag/religion-of-peace/|/tag/islamic/|/tag/shariah/|/tag/shariah-law/|/tag/wahhabism/|/tag/muslim-refugees/|/tag/islamic-center-of-davis/|/tag/islamic-education-center-of-orange-county/|/tag/islamic-law/|/tag/shia-muslims/|/tag/radical-islamic-terrorism/|/tag/islamic-terrorists/|/tag/islamic-reformation/|/tag/islam-in-europe/|/tag/iftar/|/tag/islamic-violence/|/tag/islamic-terror/|/tag/islam-in-britain/|/tag/mohammed-art-exhibit-and-contest/|/tag/islamic-fundamentalism/|/tag/muslim-ban/|/tag/sunni-islam/|/tag/muslim-violence/|/tag/islamic-faith/|/tag/islamic-jihad/|/tag/islamist-ideology/|/tag/muslim-student-association/|/tag/global-jihad-movement/|/tag/burkini/|/tag/muslim-women/|/tag/salafist/|/tag/wahabbi/|/tag/muslim-migrants/|/tag/holy-quran/|/tag/burka/|/tag/salafism/|/tag/shahada/|/tag/prophet-mohammed/|/tag/prophet-muhammad/|/tag/muslim-world/|/tag/ayatollah/|/tag/al-azhar/|/tag/hajj-pilgrimage/|/tag/islam-in-uk/|/tag/fatwas/|/tag/burqa-ban/|/tag/halal-slaughter/|/tag/sunni-muslims/|/tag/burkini-ban/|/tag/islamist-terrorism/|/tag/non-muslims/|/tag/muslim-students-association/|/tag/islamic-movement/|/tag/muslim-countries/|/tag/islamization/|/tag/criticism-of-islam/|/tag/muhammad-art-contest/'), 'muslim_identity'] = 1

	paragraphs = pd.read_csv("../data/breitbart_paragraphs.csv", dtype={"article_id": 'str'})
	paragraphs = paragraphs.loc[paragraphs.text.notnull(),]

	def get_par(a_id):
		print(a_id)
		pars = paragraphs.loc[paragraphs.article_id == a_id, 'text']
		return('\n'.join(pars))

	article_ids = articles['article_id']
	with Pool(8) as p:
		articles['text'] = p.map(get_par, article_ids)

	articles.to_csv('../data/breitbart_data.csv', index=False)
else:

	data = pd.read_csv('../data/breitbart_data.csv')

	stemmer = SnowballStemmer(language='english')
	lemmatizer = WordNetLemmatizer()

	def get_wordnet_pos(treebank_tag):
		if treebank_tag.startswith('J'):
			return wordnet.ADJ
		elif treebank_tag.startswith('V'):
			return wordnet.VERB
		elif treebank_tag.startswith('N'):
			return wordnet.NOUN
		elif treebank_tag.startswith('R'):
			return wordnet.ADV
		else:
			return None

	def split_tokenizer(text):
		return text.split()

	def token(text):
		return ' '.join(word_tokenize(text))

	def stem(text):
		tokens = word_tokenize(text)
		tokens = [stemmer.stem(t) for t in tokens]
		return ' '.join(tokens)

	def lemma(text):
		tokens = word_tokenize(text)
		lemmas = []
		for w, p in pos_tag(tokens):
			p = get_wordnet_pos(p)
			if p is None:
				lemmas.append(lemmatizer.lemmatize(w))
			else:
				lemmas.append(lemmatizer.lemmatize(w,p))
		return ' '.join(lemmas)

	data = pd.read_csv('../data/breitbart_data.csv')
	if 'stem' not in data.columns:
		with Pool(8) as p:
			data['stem'] = p.map(stem, data['text'])
		data.to_csv('../data/breitbart_data.csv', index=False)
	if 'lemma' not in data.columns:
		with Pool(8) as p:
			data['lemma'] = p.map(lemma, data['text'])
		data.to_csv('../data/breitbart_data.csv', index=False)
