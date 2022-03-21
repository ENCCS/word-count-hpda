from wordcount import load_word_counts
import numpy as np
import sys

def top_n_words(counts, number_of_top_words=10):
    """
    Given a list of (word, count, percentage) tuples,
    return the top two word counts.
    """
    limited_counts = counts[0:number_of_top_words]
    count_data = [count for (_, count, _) in limited_counts]
    return count_data

def fit_linear_loglog(row):
    X = np.log(np.arange(len(row)) + 1.0)
    ones = np.ones(len(row))
    A = np.vstack((X, ones)).T
    Y = np.log(row)
    res = np.linalg.lstsq(A, Y, rcond=-1)
    return res[0][0]

if __name__ == '__main__':
    number_of_top_words = sys.argv[1]
    input_files = sys.argv[2:]
    header = "Book"
    for i in range(int(number_of_top_words)):
         header += ",word"+str(i+1)

    header += ",power law exponent"
    print(header)
    for input_file in input_files:
        counts = load_word_counts(input_file)
        top_words = top_n_words(counts, int(number_of_top_words))
        exponent = fit_linear_loglog(top_words)
        bookname = input_file[:-4].split("/")[-1]
        line = bookname
        for i in top_words:
            line += ","+str(i)
        line += ","+str(exponent)
        print(line)
