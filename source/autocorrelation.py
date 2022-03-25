import sys
import numpy as np
from wordcount import load_word_counts, load_text, DELIMITERS
import time
from mpi4py import MPI


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

    # distribute words among MPI tasks
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
    count = nwords // n_ranks
    remainder = nwords % n_ranks
    # first 'remainder' ranks get 'count + 1' tasks each
    if rank < remainder:
        first = rank * (count + 1)
        last = first + count + 1
    # remaining 'nwords - remainder' ranks get 'count' task each
    else:
        first = rank * count + remainder
        last = first + count 

    my_words = top_words[first:last]
    print(f"My rank number is {rank} and words = {my_words}")
    # number of "timesteps" to use in autocorrelation function
    timesteps = 100
    # compute average autocorrelation and time the execution
    t0 = time.time()
    my_acf_ave = word_autocorr_average(my_words, clean_text, timesteps=100)

    receive_message = MPI.COMM_WORLD.gather(my_acf_ave, root=0)

    if rank == 0:
#        acf_ave = np.average(acf, axis=0)
        #for i in range(n_ranks):
        print(receive_message)
    sys.exit()
    t1 = time.time()        

    print(f"serial time: {t1-t0}")

    np.savetxt(sys.argv[3], np.vstack((np.arange(1,101), acf_ave)).T, delimiter=',')
