#A MEMM POS tagger for ASL

import sys
import analysis.signstream as ss
import glob
import random
from collections import defaultdict
from math import log, exp
from nltk.classify import MaxentClassifier

#a Class used for tokenizing a SignStream Annotation
class Token:
	def __init__(self,mg,ndg,pos,pos2):
		self.main_gloss = mg
		if ndg == []:
			self.nondom_gloss = "null"
		else:
			self.nondom_gloss = ndg
		if pos == [] or len(pos) != 1:
			self.pos = "null"
		else:
			self.pos = pos[0]
		if pos2 == [] or len(pos2) != 1:
			self.pos2 = "null"
		else:
			self.pos2 = pos2[0]
		self.rtoken = ((self.main_gloss,self.nondom_gloss),(self.pos,self.pos2))
		if self.pos != self.pos2 and self.pos2 != "null":
			self.not_matching = True
		else:
			self.not_matching = False
		self.features = []
		self.feature_form = {}

	def __eq__(self, other):
		if self.rtoken == other.rtoken:
			return True
		else:
			return False

	def print_token(self):
		print "token: " + str(self.rtoken)
		print self.features
	#maybe just overwrite the print method? but how


null_token = Token("","","","")
null_token2 = Token("","","","")
unknown_token = ("<UNK>","<UNK>")  # unknown word token.
start_token = ("<S>","<S>")  # sentence boundary token.
end_token = ("</S>","</S>")  # sentence boundary token.

#build a static vocabulary for preprocessing
def BuildVocab(training):
	occur = defaultdict(float)
	s_occur = set()
	cutoff = 1
	for sent in training:
		for token in sent:
			if (occur[token.rtoken[0]] >= cutoff):
				s_occur.add(token.rtoken[0])
			occur[token.rtoken[0]] += 1
	s_occur.add(unknown_token)	
	return s_occur
#add start and end tokens, as well as unknown tokens
def PreprocessText(corpus, vocabulary):
	new_corp = []
	unknown_count = 0
	total_count = 0
	for sent in corpus: 
		new_sent = []
		starter = null_token
		starter.rtoken = (start_token , start_token)
		starter.features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		new_sent.append(starter)
		for token in sent:
			total_count += 1
			if (token.rtoken[0] in vocabulary):
				new_sent.append(token)
			if (token.rtoken[0] not in vocabulary):
				new_tok = token
				new_tok.rtoken = (unknown_token,token.rtoken[1])
				new_sent.append(new_tok)
				unknown_count += 1
		ender = null_token2
		ender.rtoken = (end_token , end_token)
		ender.features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		new_sent.append(ender)
		new_corp.append(new_sent)
	#print "there are " + str(unknown_count) + " unknown tokens out of " + str(total_count)
	return new_corp

#used for formatting signstream annotations
def format_field(head_movements):
  temp = []
  for hm in head_movements:
	hstext = hm.get_text()
	temp.append(hstext)
  return temp

#identifies the features of a token based on manual and nonmanual features
def Featurize(token):
	features = []
	#the following are all manual features
	feature = []
	feature = token.get_text()
	if "fs-" in feature:
		features.append(1)
	else:
		features.append(0)

	if feature[-1] == "+":
		features.append(1)
	else:
		features.append(0)
	if feature[-2] == "+":
		features.append(1)
	else:
		features.append(0)
	if ":" in feature:
		features.append(1)
	else:
		features.append(0)
	if "IX" in feature:
		features.append(1)
	else:
		features.append(0)
	if "POSS" in feature:
		features.append(1)
	else:
		features.append(0)
	if "SELF" in feature:
		features.append(1)
	else:
		features.append(0)
	if "arc" in feature:
		features.append(1)
	else:
		features.append(0)
	if "loc" in feature:
		features.append(1)
	else:
		features.append(0)
	if "dir" in feature:
		features.append(1)
	else:
		features.append(0)
	if "continuative" in feature:
		features.append(1)
	else:
		features.append(0)
	if "recip" in feature:
		features.append(1)
	else:
		features.append(0)

	if len(token.get_coinciding_tokens("nd hand gloss")) > 0:
		nd = token.get_coinciding_tokens("nd hand gloss")[0].get_text()
		if feature == nd:
			features.append(1) #same
			features.append(0) #different
			features.append(0) #none
		if feature != nd:
			features.append(0) #same
			features.append(1) #different
			features.append(0) #none
		else:
			features.append(0) #same
			features.append(0) #different
			features.append(1) #none

	#the following are all nonmanual features
	feature = []
	feature = token.get_coinciding_tokens("negative")
	if feature == []:
		features.append(0)
	elif len(feature) > 0:
		for x in feature:
			if x.get_text() == "neg" or x.get_text() == "negation":
				features.append(1)

	feature = []
	feature = token.get_coinciding_tokens("wh question")
	if feature == []:
		features.append(0)
	elif len(feature) > 0:
		for x in feature:
			if x.get_text() == "whq":
				features.append(1)

	feature = []
	feature = token.get_coinciding_tokens("yes-no question")
	if feature == []:
		features.append(0)
	elif len(feature) > 0:
		for x in feature:
			if x.get_text() == "yes-no question":
				features.append(1)

	feature = []
	feature = token.get_coinciding_tokens("cond/when")
	if feature == []:
		features.append(0)
		features.append(0)
	elif len(feature) > 0:
		for x in feature:
			if x.get_text() == "conditional":
				features.append(1)
				features.append(0)
			if x.get_text() == "when":
				features.append(0)
				features.append(1)

	feature = []
	feature = token.get_coinciding_tokens("rhq")
	if feature == []:
		features.append(0)
		features.append(0)
		features.append(0)
	elif len(feature) > 0:
		for x in feature:
			if x.get_text() == "rhq":
				features.append(1)
				features.append(0)
				features.append(0)
			if x.get_text() == "wh rhq":
				features.append(0)
				features.append(1)
				features.append(0)
			if x.get_text() == "yes-no rhq":
				features.append(0)
				features.append(0)
				features.append(1)

	feature = []
	feature = token.get_coinciding_tokens("topic/focus")
	if feature == []:
		features.append(0)
	elif len(feature) > 0:	#we only want to use the nonmanual feature, not the annotator's knowledge of the rest of the sentence
		features.append(1)	#do deeper analysis of this if time left

	feature = []
	feature = token.get_coinciding_tokens("role shift")
	if feature == []:
		features.append(0)
	elif len(feature) > 0:
		features.append(1)

	return features

#this function reads in an XML file containing SignStream Annotations
#and tokenizes the signs to create the corpus
#returns an array of sentences
def add_to_corpus(file):
	db = ss.SignStreamDatabase.read_xml(file)
	fields = db.get_fields()
	sentences = []
	for participant in db.get_participants():
		for utterance in participant.get_utterances():
			sent = []
			for token in utterance.get_tokens_for_field("main gloss"):
				text = token.get_text()
				(start, end) = token.get_timecodes()

				nondm = format_field( token.get_coinciding_tokens("nd hand gloss") )
				if len(nondm) != 0:
					nondmm = nondm[0]
					(start, end) = token.get_coinciding_tokens("nd hand gloss")[0].get_timecodes()

				else:
					nondmm = nondm

				pos = format_field(token.get_coinciding_tokens("POS"))
				if len(pos) != 0:
					poss = pos[0][0]
				pos2 = format_field(token.get_coinciding_tokens("POS2"))
				new_token = Token(text,nondmm,pos,pos2)
				new_token.features = Featurize(token)
				if new_token.rtoken[1][0] == new_token.rtoken[1][1]:	#dont double tag two handed signs
					new_token.rtoken = (new_token.rtoken[0], (new_token.rtoken[1][0],'null'))
				if new_token.rtoken[1] ==  ('null', 'null'):	#don't use nonlinguistic gestures
					new_token.rtoken = (('null', 'null'),('null', 'null'))
				sent.append(new_token)

			sentences.append(sent)
	return sentences


#this is the actual tagger
#initialize it with a maxent classifier fed with your training data
class MEMM_tagger:
	def __init__(self,maxent):
		self.ttcounts = defaultdict(float)
		self.tagcounts = defaultdict(float)
		self.transition = defaultdict(lambda: log(1.0/58))

		self.wordcounts = defaultdict(float)
		self.wtcounts = defaultdict(float)
		self.emissions = defaultdict(float)

		self.dictionary = defaultdict(set) #maps word to possible set of tags
		self.tagset = set()
		self.maxent = maxent

	#trains a bigram HMM
	def train(self,training_set):
		last_tok = Token("","","","")
		#take wt tt w and t counts
		for sent in training_set:
			last_tok = Token("","","","")
			for tok in sent:
				real_token = tok.rtoken[0]
				real_tag = tok.rtoken[1]
				if real_tag not in self.tagset:
					self.tagset.add(real_tag)
				self.wtcounts[ (tok.rtoken[0], real_tag) ] += 1.0 #we need to store the whole tokens so we can get the features later
				self.tagcounts[tok.rtoken[1]] += 1.0
				if tok.rtoken[1] not in self.dictionary[tok.rtoken[0]]: #if tag not on list of tags for this token add it
					self.dictionary[tok.rtoken[0]].add(tok.rtoken[1])
				if (last_tok != Token("","","","")):
					self.ttcounts[(last_tok.rtoken[1],tok.rtoken[1])] += 1.0
		print self.tagset
		print len(self.tagset)
		tag_len = len(self.tagset)
		self.transition = defaultdict(lambda: log(1.0/tag_len))
		for k, v in self.ttcounts.iteritems():
			self.transition[k] = log( (v +1) / (self.tagcounts[k[0]] +tag_len) )
		#estimate B matrix (word given its tag)
		for k, v in self.wtcounts.iteritems():
			self.emissions[k] = log( v / self.tagcounts[k[1]] )

	#tags a sentence using the bigram HMM
	def hmmViterbi(self, sent):
		tagged_sent = defaultdict(str)
		viterbi = defaultdict(float)
		backpointers = {}  #keys are states values are other states

		#handle the first step
		for tag in self.dictionary[sent[1].rtoken[0]]:
			viterbi[(tag,1)] = self.transition[(sent[0].rtoken[1], tag)] + self.emissions[(sent[1].rtoken[0], tag)]
			backpointers[(tag,1)] = ((start_token), 0)
		viterbi[((start_token), 0)] = 0

		#recursion step
		counter = 1
		for token in sent[2:]:
			counter += 1
			for tag in self.dictionary[token.rtoken[0]]:
				find_max = defaultdict(float)
				for state in viterbi:
					if (state[1] == counter-1):
						find_max[state]= viterbi[state] + self.transition[(state[0],tag)] + self.emissions[(token.rtoken[0],tag)]
				if len(find_max.values()) >= 1:
					viterbi[(tag, counter)] = max(find_max.values())
					backpointers[(tag, counter)] = max(find_max, key=lambda i: find_max[i])

		#termination step should be included in the recursion
		#traverse the backtrace and tag the sentence
		lookstate = (end_token,len(sent)-1)
		for k in backpointers.keys():
				tagged_sent[backpointers[lookstate][1]] = backpointers[lookstate][0]
				if backpointers[lookstate] != ((start_token), 0):
					lookstate = backpointers[lookstate]
		tagged_sent[0] = (start_token)

		final_tagged = []
		for count, k in enumerate(tagged_sent.values()):
			final_tagged.append((sent[count].rtoken[0], k))
		final_tagged.append((end_token,end_token))
		return final_tagged

	#tags a sentence using the MEMM - notice it does not use joint probabilities
	def memmViterbi(self, sent):
		tagged_sent = defaultdict(str)
		viterbi = defaultdict(float)
		backpointers = {}  #keys are states values are other states

		#handle the first step
		for tag in self.dictionary[sent[1].rtoken[0]]:
			viterbi[(tag,1)] = self.maxent.prob_classify(sent[0].feature_form).logprob(tag)
			backpointers[(tag,1)] = ((start_token), 0)
		viterbi[((start_token), 0)] = 0

		#recursion step
		counter = 1
		for token in sent[2:]:
			counter += 1
			for tag in self.dictionary[token.rtoken[0]]:
				find_max = defaultdict(float)
				for state in viterbi:
					if (state[1] == counter-1):
						find_max[state]= viterbi[state] + self.maxent.prob_classify(token.feature_form).logprob(tag)
				if len(find_max.values()) >= 1:
					viterbi[(tag, counter)] = max(find_max.values())
					backpointers[(tag, counter)] = max(find_max, key=lambda i: find_max[i])

		#termination step included in the recursion
		#traverse the backtrace and tag the sentence
		lookstate = (end_token,len(sent)-1)
		for k in backpointers.keys():
				tagged_sent[backpointers[lookstate][1]] = backpointers[lookstate][0]
				if backpointers[lookstate] != ((start_token), 0):
					lookstate = backpointers[lookstate]
		tagged_sent[0] = (start_token)

		final_tagged = []
		for count, k in enumerate(tagged_sent.values()):
			final_tagged.append((sent[count].rtoken[0], k))
		final_tagged.append((end_token,end_token))
		return final_tagged

	#run hmmViterbi on the entire test set
	def Test(self, test_set):
		""" Use Viterbi and predict the most likely tag sequence for every sentence. Return a re-tagged test_set. """
		tagged_set = []
		for sent in test_set:
			tagged_set.append(self.hmmViterbi(sent))
		return tagged_set

	#run memmViterbi on the entire test set
	def memmTest(self, test_set):
		tagged_set = []
		for sent in test_set:
			tagged_set.append(self.memmViterbi(sent))
		return tagged_set

	#run the most common class baseline on the entire test set
	def most_common_class_baseline(self,test_set):
		#tag the test by just looking at wtcounts
		tagged_test = []
		for sent in test_set:
			new_sent = []
			for tok in sent:
				#find most common tag and make a new token same as the old one but with new tag and append to sent
				temp = defaultdict(float)
				for tag in self.dictionary[tok.rtoken[0]]:
					temp[(tok.rtoken[0],tag)] = self.wtcounts[(tok.rtoken[0],tag)]
				new_token = max(temp, key=lambda i: temp[i])
				new_sent.append(new_token)
			tagged_test.append(new_sent)
		return tagged_test

	#compute the percent number of tokens which can have multiple tags
	def ComputePercentAmbiguous(self, data_set):
		""" Compute the percentage of tokens in data_set that have more than one tag according to self.dictionary. """
		num_ambig = 0.0
		num_toks = 0.0
		for sent in data_set:
			for tok in sent:
				num_toks +=1.0
				if len(self.dictionary[tok.rtoken[0]]) > 1:
					num_ambig += 1.0
			num_toks += -2.0
		return 100*num_ambig/num_toks

#returns a confusion table given the gold standard tags and tagger's output
def Confusion(test_set, test_set_predicted):
	confusion = defaultdict(float) #keys are incorrect comma correct tokens
	for count, sent in enumerate(test_set):
		for t_count, token in enumerate(sent):
			if token.rtoken != test_set_predicted[count][t_count]:
				confusion[(token.rtoken[1], test_set_predicted[count][t_count][1])] += 1
	return confusion

#prints accuracy percentages for your tagger given the gold standard tags
def ComputeAccuracy(test_set, test_set_predicted):
	""" Using the gold standard tags in test_set, compute the sentence and tagging accuracy of test_set_predicted. """
	correct_sentences = 0.0
	correct_tags = 0.0
	num_toke = 0.0
	for count, sent in enumerate(test_set):
		sent_same = True
		for t_count, token in enumerate(sent):
			num_toke += 1
			if token.rtoken[1] == test_set_predicted[count][t_count][1]:
				correct_tags +=1
			else:
				sent_same = False
		if sent_same:
			correct_sentences += 1
		correct_tags += -2
		num_toke += -2
	print "Percent sentence accuracy: " + str(100*correct_sentences/len(test_set))
	print "Percent tag accuracy: " + str(100*correct_tags/num_toke)
	return ( 100*correct_sentences/len(test_set), 100*correct_tags/num_toke )

def main():
	corpus = []
	#grab the corpus from its directory
	files = glob.glob("../ncslgr-xml/*.xml")
	#build the corpus and tokenize the data using signstream parser
	for file in files:
		corpus += add_to_corpus(file)
	for token in corpus[672]:
		token.print_token()
	print len(corpus)

	#shuffle the corpus to account for differences in genres of elicited data
	random.shuffle(corpus)
	#dev_set = corpus[:1698]
	#test_set = corpus[189:]
	dev_set = corpus[:300]
	test_set = corpus[300:600]
	#build a static vocabulary
	vocab = BuildVocab(dev_set)
	#add start,end, and unknown tokens
	preprocessed_dev_set = PreprocessText(dev_set,vocab)
	preprocessed_test_set = PreprocessText(test_set,vocab)
	#print the first sentence of the Training Set
	for token in preprocessed_dev_set[0]:
		token.print_token()

	#add word identity features to the Training Set
	for sent in preprocessed_dev_set:
		for token in sent:
			for word in vocab:
				if token.rtoken[0] == word:
					token.features.append(1)
				else:
					token.features.append(0)
	#keep track of which tags are in the training set
	tagset = set()
	for sent in preprocessed_dev_set:
		for tok in sent:
			if tok.rtoken[1] not in tagset:
				tagset.add(tok.rtoken[1])

	#add word identity tokens to the Test Set
	for sent in preprocessed_test_set:
		for token in sent:
			for word in vocab:
				if token.rtoken[0] == word:
					token.features.append(1)
				else:
					token.features.append(0)

	#format Training and Test set embeddings for NLTK's MaxEnt
	for sent in preprocessed_dev_set:
		for tok in sent:
			count = 0
			for feat in tok.features:
				if feat != 0:
					tok.feature_form[str(count)] = feat
				count += 1
	training_data = []
	for sent in preprocessed_dev_set:
		for tok in sent:
			training_data.append((tok.feature_form,tok.rtoken[1]))

	for sent in preprocessed_test_set:
		for tok in sent:
			count = 0
			for feat in tok.features:
				if feat != 0:
					tok.feature_form[str(count)] = feat
				count += 1
	test_data = []
	for sent in preprocessed_test_set:
		for tok in sent:
			test_data.append(tok.feature_form)

	#initialize the tagger with an NLTK maxent classifier
	mem = MEMM_tagger(MaxentClassifier.train(training_data,"GIS"))
	#Count Bigrams for the Training Set
	mem.train(preprocessed_dev_set)
	#Run the Most Common Class Baseline and print accuracy reports
	print "MCC Baseline accuracies:"
	common = mem.most_common_class_baseline(preprocessed_test_set)
	ComputeAccuracy(preprocessed_test_set,common)
	#show the first sentence as tagged by the MCC Baseline
	print "First sentence tagged by MCC baseline:"
	confusion = Confusion(preprocessed_test_set,common)
	print common[0]
	#show the most common mistakes made
	print "Most common mistakes:"
	print max(confusion.values())
	print max(confusion, key=lambda i: confusion[i])
	#save the confusion table to another file
	target = open("MCC_confusion", 'w')
	target.write(str(confusion))
	target.close()

	#run the Viterbi algorithm on the Test set using an HMM
	predicted = mem.Test(preprocessed_test_set)
	print "first sentence as tagged by bigram HMM:"
	print predicted[0]
	ComputeAccuracy(preprocessed_test_set,predicted)
	#report accuracy and confusion for HMM
	print "Most common mistakes:"
	confusion = Confusion(preprocessed_test_set,predicted)
	print max(confusion.values())
	print max(confusion, key=lambda i: confusion[i])
	target = open("HMM_confusion", 'w')
	target.write(str(confusion))
	target.close()

	#run viterbi on the test set using MEMM
	predicted = mem.memmTest(preprocessed_test_set)
	print "first sentence as tagged by unigram MEMM:"
	print predicted[0]
	ComputeAccuracy(preprocessed_test_set,predicted)
	#report accuracy and confusion for MEMMM
	confusion = Confusion(preprocessed_test_set,predicted)
	print max(confusion.values())
	print max(confusion, key=lambda i: confusion[i])

	print "Vocab size: " + str(len(vocab))

	target = open("MEMM_confusion", 'w')
	target.write(str(confusion))
	target.close()

	print "percent ambiguity in the training set:"
	print mem.ComputePercentAmbiguous(preprocessed_dev_set)
	
if __name__ == "__main__": 
	main()