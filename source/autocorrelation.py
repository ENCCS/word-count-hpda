import sys
import numpy as np
from wordcount import load_word_counts, load_text, DELIMITERS
import time
from multiprocessing import Pool
from functools import partial

def preprocess_text(text):
    """
    Remove delimiters, split lines into words and remove whitespaces, 
    and make lowercase. Return list of all words in the text.
    """
    clean_text = []
    for line in text:
        for purge in DELIMITERS:
            line = line.replace(purge, " ")    
        words = line.split()
        for word in words:
            word = word.lower().strip()
            clean_text.append(word)
    return clean_text

def word_autocorr(word, text, timesteps):
    """
    Calculate word-autocorrelation function for given word 
    in a text. Each word in the text corresponds to one "timestep".
    """
    acf = np.zeros((timesteps,))
    mask = [w==word for w in text]
    nwords_chosen = np.sum(mask)
    nwords_total = len(text)
    for t in range(timesteps):
        for i in range(1,nwords_total-t):
            acf[t] += mask[i]*mask[i+t]
        acf[t] /= nwords_chosen      
    return acf
    
def word_autocorr_average(words, text, timesteps=100):
    """
    Calculate an average word-autocorrelation function 
    for a list of words in a text.
    """
    acf = np.zeros((len(words), timesteps))
    for n, word in enumerate(words):
        acf[n, :] = word_autocorr(word, text, timesteps)
    return np.average(acf, axis=0)

def word_autocorr_average_pool(words, text, timesteps=100):
    """
    Calculate an average word-autocorrelation function 
    for a list of words in a text using multiprocessing.
    """
    word_autocorr_partial = partial(word_autocorr, text=text, timesteps=timesteps)
    with Pool(4) as p:
        results = p.map(word_autocorr_partial, words)
    acf = np.array(results)
    return np.average(acf, axis=0)

if __name__ == '__main__':
    # load book text and preprocess it
    book = sys.argv[1]
    text = load_text(book)
    clean_text = preprocess_text(text)
    # load precomputed word counts and select top 10 words
    wc_book = sys.argv[2]
    nwords = 10
    word_count = load_word_counts(wc_book)
    top_words = [w[0] for w in word_count[:nwords]]
    # number of "timesteps" to use in autocorrelation function
    timesteps = 100
    # compute average autocorrelation and time the execution
    t0 = time.time()
    acf_ave = word_autocorr_average(top_words, clean_text, timesteps=100)
    t1 = time.time()        
#    acf_pool_ave = word_autocorr_average_pool(top_words, clean_text, timesteps=100)
    t2 = time.time()        
    print(f"serial time: {t1-t0}")
    print(f"parallel map time: {t2-t1}")

    np.savetxt(sys.argv[3], np.vstack((np.arange(1,101), acf_ave)).T, delimiter=',')
    #sys.exit()
#    np.testing.assert_array_equal(acf_ave, acf_pool_ave)
#    print(acf_ave-acf_pool_ave)
    #plt.plot(acf_ave)
    #plt.plot(acf_pool_ave)
    #plt.ylim([0.0,0.3])
    #plt.savefig('tmp.png')