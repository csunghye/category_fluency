# Category fluency pipeline

This pipeline processes audio files of category-guided fluency tasks, automatically calculates scores and rates several language measures per word (or per word pair for semantic and phonetic similarities). It requires audio files and corresponding forced aligned outputs when processing. After processing, it generates two comma-separated files: one file for automatically calculated scores and the other for language measures per word. To run the program, enter a command like the following:

`python3 category_fluency_pipeline.py -score_file score.csv -measure_file results.csv -audio_folder /audio_location -FA_folder /FA_location -category animal  -glove_folder /glove`

## Prerequisites

- GloVe: (https://nlp.stanford.edu/projects/glove/) 
- Python packages: `spacy`, `scipy`, `pydub`, `py-dtw`, `librosa`, `numpy`, `pandas`, `nltk`, `textblob`, `gensim`

The 6B.300d GloVe word embedding file is used to calculate semantic distance between words, and the large language model ('en_core_web_lg') from spaCy is used for POS tagging. Please make sure that users have all the packages installed before running the program.

## Arguments and options

- `-score_file`: This is a required argument for the output file name for the score file.   
- `-measure_file`: This is also a required argument for the output file name for the measure file. 
- `-audio_folder`: This is also a required argument for the folder name where audio files are. 
- `-FA_folder`: This is also a required argument for the folder name where forced alignment results are. Note that the alignment files should be **tab-separated** and need to include pause duration between words.  
- `-FA_filetype`: This is an optional argument (type: `str`) and users can indicate their FA output filetype (e.g., `*.txt`). If not entered, the default `*.word` will be used. Note that users cannot use Praat textgrids for FA outputs.
- `-category`: This is a required argument. Please enter the category of the fluency task, e.g., animal.
- `-glove_folder`: This is also a required argument for the location of GloVe word embedding files. 
- `-speaker`: 

## Measurements
This pipeline measures the following:

### Score file
- Number of total words: Total number of words starting with a given letter including proper nouns, numerics, and repetitions
- Number of proper nouns: The number of proper nouns starting with a given letter (e.g., "Fanta" for "f")
- Number of numerics: The number of numeric words starting with a given letter (e.g., "forty" for "f")
- Number of repetitions: The number of repeated answers starting with a given letter. This count is based on lemma of words, so "flower" and "flowers" are considered as a repetition.  

So, the actual score of a speaker would be total number of words - (number of proper nouns + numerics + repetitions). 

### Language measure file
- Previous pause duration: The pause duration between the previous correct response and the current correct response. This includes the duration of partial words, fillers, and all other words that do not belong to the given category (e.g., "Wow, this is difficult!").
- Word duration: The duration of correct responses. 
- Number of syllables: The number of syllables of correct responses from the CMU dictionary
- Several lexical measures of correct responses: Age of acquisition (AoA), word frequncy, word familiarity, semantic ambiguity, concreteness: These measures are based on published norms. See [Citation](#citation)
- Semantic distance between the previous correct response and the current correct response: This measure is calculated with GloVe word embeddings.
- Phonetic distance between the previous correct response and the current correct response: Phonetic similarity is measured by extracting 1st to 13th MFCC values of the previous and the currect correct responses, applying dynamic time warping to normalize duration, and calculating an Euclidean distance between two time-warped matrices. 

## Citation
- When using this pipeline, please cite this paper: Cho, Sunghye, Naomi Nevler, Natalia Parjane, Christopher Cieri, Mark Y. Liberman, Murray Grossman, Katheryn A. Q. Cousins. (2021). [Automated analysis of digitized letter fluency data.](https://www.frontiersin.org/articles/10.3389/fpsyg.2021.654214/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Psychology&id=654214) Frontiers in Psychology 12, 654214.
- To cite the lexical measures used:
    - Word frequency: Brysbaert, M., and New, B. (2009). Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms and the introduction of a new and improved word frequency measure for American English. Behav. Res. Methods 41, 977–990. doi: 10.3758/BRM.41.4.977
    - Word familarity, AoA: Brysbaert, M., Mandera, P., and Keuleers, E. (2018). Word prevalence norms for 62,000 English lemmas. Behav. Res. Methods 51, 467–479. doi: 10.3758/s13428-018-1077-9
    - Semantic ambiguity: Hoffman, P., Lambon Ralph, M. A., and Rogers, T. T. (2013). Semantic diversity: a measure of semantic ambiguity based on variability in the contextual usage of words. Behav. Res. Methods 45, 718–730. doi: 10.3758/s13428-012-0278-x
    - Concreteness: Brysbaert, M., Warriner, A. B., and Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. Behav. Res. Methods 46, 904–911. doi: 10.3758/s13428-013-0403-5
    - Number of syllables: Carnegie Mellon Speech Group (2014). The Carnegie Mellon University Pronouncing Dictionary. Available at: http://www.speech.cs.cmu.edu/cgi-bin/cmudict (Accessed May 25, 2020).


## Note
I've seen many people using a cluster approach, but this hasn't been implemented here yet. I will explore cluster approaches tried in previous studies later and implement one in this pipeline. 
