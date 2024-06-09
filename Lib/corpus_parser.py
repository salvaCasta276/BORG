import os
import csv
import numpy as np
from os.path import isfile, join

number_of_texts = 13

#Transforma una coleccion de csvs contenidos en un directorio en una lista de bloques de palabras
#El tag esta ordenado en base al orden alfabetico de los nombres de los archivos
def parse_corpus(dir_path, block_size):

    drct = os.listdir(dir_path)
    dataset = np.ones(block_size+1).reshape((1, block_size+1))
    drct.sort()

    for j, file_name in enumerate(drct):
        file_path = join(dir_path, file_name)

        if not isfile(file_path):
            continue
        file = open(file_path)
        reader = csv.reader(file)
        next(reader)
        for i in range(number_of_texts):
            words = next(reader)[2].split()
            while len(words) % block_size != 0:
                words.append("padding")
            words = np.array(words)
            words = words.reshape((int(len(words)/block_size), block_size))
            words = np.append(words, j*np.ones(len(words)).reshape(len(words), 1), axis=1)
            dataset = np.append(dataset, words, axis=0)
    return dataset, drct

