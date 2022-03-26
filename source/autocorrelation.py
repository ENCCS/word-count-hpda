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
    
    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # distribute words among MPI tasks
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
    # each rank gets unique words
    my_words = top_words[first:last]
    print(f"My rank number is {rank} and first, last = {first}, {last}")

    # number of "timesteps" to use in autocorrelation function
    timesteps = 100

    # only rank 0 will collect all acfs
    if rank == 0:
        acf_all = np.zeros((nwords, timesteps))
    # each rank computes own set of acfs
    my_acfs = np.zeros((len(my_words), timesteps))
    for i, word in enumerate(my_words):
        my_acfs[i,:] = word_autocorr(word, clean_text, timesteps)

    # rank 0 receives data from other ranks
    if rank == 0:
        # first copy own my_acfs to acf_all
        for n, i in enumerate(range(first, last)):
            acf_all[i,:] = my_acfs[n,:]
        # then receive from other workers
        for sender in range(1, n_ranks):
            # first receive indices
            rec_first, rec_last = comm.recv(source=sender, tag=10)
            # then receive data
            acf_all[rec_first:rec_last,:] = comm.recv(source=sender, tag=12)
    else:
        # first send indices
        comm.send([first, last], dest=0, tag=10)
        # then send data
        comm.send(my_acfs, dest=0, tag=12)

    # rank 0 computes average and saves result
    if rank == 0:
        acf_ave = np.average(acf_all, axis=0)
        np.savetxt(sys.argv[3], np.vstack((np.arange(1,101), acf_ave)).T, delimiter=',')



