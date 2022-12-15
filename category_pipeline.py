## Usage python3 animals_pipeline.py -measurement_file all_results.csv -score_file scores.csv -input_folder ~/Documents/speechbiomarkers/fluency/animals/subset_carmen -category animal

import spacy
import argparse, glob
from scipy import spatial
from pydub import AudioSegment
#from pydub.playback import play
from dtw import *
import librosa
import numpy as np
import pandas as pd
import gensim.models 
from textblob import Word
from nltk.corpus import wordnet as wn


#animal = Word("animal").synsets[0]

#def get_hyponyms(synset):
#    hyponyms = set()
#    for hyponym in synset.hyponyms():
#        hyponyms |= set(get_hyponyms(hyponym))
#    return hyponyms | set(synset.hyponyms())

#animal_list = get_hyponyms(animal)


nlp = spacy.load('en_core_web_lg')
LEXICAL_LOOKUP = './all_measures_raw.csv'
FILLERS = ['um','uh','eh', 'oh']
BACKCHANNELS = ['hm', 'yeah', 'mhm', 'huh']
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/csunghye/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz', binary=True)


def get_hyponyms(synset):
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms(hyponym))
    return hyponyms | set(synset.hyponyms())

def correct_word(word, args):
	animal = Word(args.category).synsets[0]
	correct_list = get_hyponyms(animal)
	if (set(Word(word).synsets) & set(correct_list)):
		return True
	else:
		return False

def get_wordVec(WORDVEC):
    f = open(WORDVEC,'r')
    embeddings_dict = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = [value for value in splitLines[1:]]
        embeddings_dict[word] = wordEmbedding
    return embeddings_dict

def get_mfcc(data_wav, start, end, fs):
	wav_crop = data_wav[float(start)*1000 : float(end)*1000]
	#play(wav_crop)
	wav_crop_samples = wav_crop.get_array_of_samples()
	mfcc_val = librosa.feature.mfcc(y=np.array(wav_crop_samples).astype(np.float32), sr=fs, n_mfcc=13)
	mfcc_transpose = np.transpose(mfcc_val)
	return mfcc_transpose

def read_audio(filename):
	data_wav = AudioSegment.from_wav(filename)
	fs = np.array(data_wav.frame_rate)
	return data_wav, fs

def get_phon_sim(prev_mfcc, mfcc):
	if np.isnan(prev_mfcc).all() != True :
		phon_sim = dtw(prev_mfcc, mfcc, keep_internals=True, step_pattern='asymmetricP0',window_args = {'window_size':100}).normalizedDistance
	else:
		phon_sim = np.nan
	return phon_sim

def get_sem_sim(word, prev_word):
	#sem_sim = spatial.distance.euclidean(np.asarray(embeddings_dict[word], "float32"), np.asarray(embeddings_dict[prev_word], "float32"))
	if prev_word != "NA":
		sem_sim = model.wmdistance(prev_word, word)
	else:
		sem_sim = np.nan
	return sem_sim

def get_dur_pos_phon_sem(file, data_wav, fs, correct_list, args):
    newdf = []
    prev_mfcc = np.array([], dtype=np.float64)
    infile = open(file, 'r')
    wav_name = file.split('/')[-1][:-8]+'.wav'
    pause_dur = 0
    order = 0
    prev_word = 'NA'

    for line in infile:
        if not (line.startswith('\\')):
            data = line.rstrip('\n').split('\t')
            if len(data) > 2:
                if len(data[2].split()) > 1:
                    word = data[2].split()[0]+'_'+data[2].split()[1]
                else:
                    word = data[2]
                if correct_word(word, correct_list) == True:
                    word_dur = float(data[1]) - float(data[0])
                    mfcc = get_mfcc(data_wav, data[0], data[1], fs)
                    phon_sim = get_phon_sim(prev_mfcc, mfcc)
                    sem_sim = get_sem_sim(word.lower(), prev_word)
                    order +=1
                    newdf.append([wav_name, data[0], data[1], word.lower(), order, word_dur, pause_dur, "NOUN", word.lower(), phon_sim, sem_sim])
                    prev_mfcc = mfcc
                    prev_word = word.lower()
                    word_dur = 0
                    pause_dur = 0
               
                else: 
                    if args.speaker:
                        if args.speaker == data[3]:
                            word_dur = float(data[1]) - float(data[0])
                            pause_dur += word_dur
                        else:
                            pass
                    else:
                        word_dur = float(data[1]) - float(data[0])
                        pause_dur += word_dur

    df = pd.DataFrame(newdf, columns=['file','start','end','word','order','word_dur','prev_pause_dur','POS','lemma', 'phon_sim', 'sem_sim'])
    return df

def count_correct_words(df):
	all_f_words = len(df)
	propn = df.POS.str.startswith("PROPN").sum()
	num = df.POS.str.startswith("NUM").sum()
	repetition = df.duplicated('lemma').sum()
	#print(key, repetition)
	return all_f_words, propn, num, repetition

def add_lexical(df, measureDict, phonDf):
	word_lexical = pd.merge(df, measureDict, on='word', how='left')
	lemma_lexical = pd.merge(df, measureDict, left_on='lemma', right_on='word', how='left')
	df = word_lexical.fillna(lemma_lexical)
	word_phone = pd.merge(df, phonDf[["word", "phon", "syll"]], on='word', how='left')
	lemma_phone = pd.merge(df, phonDf[["word", "phon", "syll"]], left_on='lemma', right_on='word',how='left')
	df = word_phone.fillna(lemma_phone)
	return df

def get_phondict():
	phonDf = pd.DataFrame.from_dict(nltk.corpus.cmudict.dict(), orient='index')
	phonDf = phonDf.reset_index()
	phonDf['phon'] = phonDf[0].map(len)
	phonDf = phonDf.drop(columns=[1,2,3,4])
	phonDf.columns = ['word','pron','phon']
	phonDf['pronstring'] = [','.join(map(str, l)) for l in phonDf['pron']]
	phonDf['syll'] = phonDf.pronstring.str.count("0|1|2")
	return phonDf

def main(args):
    outputname = args.measure_file
    scorename = args.score_file
    filelist = glob.glob(args.audio_folder+'/*.wav')
    category = Word(args.category).synsets[0]
    correct_list = get_hyponyms(category)
    measureDict = pd.read_csv(LEXICAL_LOOKUP)
    phonDf = get_phondict()
    allResults = pd.DataFrame()
    if args.FA_filetype:
        filelist = glob.glob(args.FA_folder+'/'+args.FA_filetype)
    else:
        filelist = glob.glob(args.FA_folder+'/*.word')
    
    with open(scorename, 'w') as outFile:
        for file in filelist:
            print(file)
            filename = file.split('.')[0]+args.FA_filetype
            data_wav,fs = read_audio(file)
            wav_name = file.split('/')[-1]
            df = get_dur_pos_phon_sem(filename, data_wav, fs, correct_list, args)
            score, propn, number,repetition = count_correct_words(df)
            df_lexical = add_lexical(df, measureDict, phonDf)
            allResults = pd.concat([allResults, df_lexical], sort=True)
            outFile.writelines(wav_name+','+str(score)+','+str(propn)+','+str(number)+','+str(repetition)+'\n')
    allResults.to_csv(outputname, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process animals fluency data')
    parser.add_argument('-measure_file', type=str, required=True, help='Name of the output file with all measures')
    parser.add_argument('-audio_folder', type=str, required=True, help='Folder containing input wav files')
    parser.add_argument('-score_file', type=str, required=True, help='Name of the output file with scores')
    parser.add_argument('-category', type=str, required=True, help='category of the semantic fluency task')
    parser.add_argument('-speaker', type=str, required=False, help='Speaker label to be analyzed')
    parser.add_argument('-FA_folder', type=str, required=True, help='Folder containing forced alignment outputs')
    parser.add_argument('-FA_filetype', type=str, required=False, help='File type of FA outputs')
    args = parser.parse_args()
    main(args)
