#!/usr/bin/python
# -*- coding:utf-8 -*-

import jieba
import os
import re
import time
import math
import numpy as np
import random
from gensim import corpora, models
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from collections import defaultdict, Counter
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def \
        data_preprocessing(data_roots):
    print("Preparing data...")
    output_path = "./corpus.txt"
    if os.path.exists(output_path):
        print("Data already prepared!")
        return
    output_file = open(output_path, 'w', encoding='utf-8')
    listdir = os.listdir(data_roots)

    char_to_be_replaced = "\n `1234567890-=/*-~!@#$%^&*()_+qwertyuiop[]\\QWERTYUIOP{}|asdfghjkl;" \
                          "'ASDFGHJKL:\"zxcvbnm,./ZXCVBNM<>?~！@#￥%……&*（）——+【】：；“‘’”《》？，。" \
                          "、★「」『』～＂□ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
                          "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
                          "ｕｖｗｙｚ￣\u3000\x1a"
    char_to_be_replaced = list(char_to_be_replaced)
    stop_words_list = []

    people_names_file = open("/data/dxy/codes/NLP_homework/NLP_homework4/Character_names.txt", "r", encoding="gbk")
    people_names = []
    for line in people_names_file.readlines():
        line = line.strip()
        jieba.add_word(line)
        people_names.append(line)

    kongfu_names_file = open("/data/dxy/codes/NLP_homework/NLP_homework4/Kongfu_names.txt", "r", encoding="gbk")
    kongfu_names = []
    for line in kongfu_names_file.readlines():
        line = line.strip()
        jieba.add_word(line)
        kongfu_names.append(line)

    sect_names_file = open("/data/dxy/codes/NLP_homework/NLP_homework4/Sect_names.txt", "r", encoding="gbk")
    sect_names = []
    for line in sect_names_file.readlines():
        line = line.strip()
        jieba.add_word(line)
        sect_names.append(line)

    print("Total number of added people, kongfu, sect:{}, {}, {}".format(len(people_names), len(kongfu_names), len(sect_names)))


    for tmp_file_name in listdir:
        print("Preprocessing <{}>...".format(tmp_file_name))
        if tmp_file_name == "inf.txt":
            continue
        path = os.path.join(data_roots, tmp_file_name)
        if os.path.isfile(path):
            with open(path, "r", encoding="gbk", errors="ignore") as tmp_file:
                tmp_file_context = tmp_file.read()
                tmp_file_lines = tmp_file_context.split("。")
                for tmp_line in tmp_file_lines:
                    for tmp_char in char_to_be_replaced:
                        tmp_line = tmp_line.replace(tmp_char, "")
                    # for tmp_char in stop_words_list:
                    #     tmp_line = tmp_line.replace(tmp_char, "")
                    tmp_line = tmp_line.replace("本书来自免费小说下载站更多更新免费电子书请关注", "")
                    if tmp_line == "":
                        continue
                    tmp_line = list(jieba.cut(tmp_line))
                    tmp_line_seg = ""
                    for tmp_word in tmp_line:
                        tmp_line_seg += tmp_word + " "
                    output_file.write(tmp_line_seg.strip() + "\n")

    output_file.close()
    print("Data succesfully prepared!")


if __name__ == '__main__':
    data_roots = '/data/dxy/codes/NLP_homework/NLP_homework_2-main/txt_files/'  # replace this path with the txt files path
    data_preprocessing(data_roots)

    ### training model
    print("Training model...")
    sentences = LineSentence('./corpus.txt')
    model_cbow = models.word2vec.Word2Vec(sentences, sg=0, vector_size=200, window=5, min_count=5, workers=8)
    model_cbow.save("./model_cbow.model")
    model_skip_gram = models.word2vec.Word2Vec(sentences, sg=1, vector_size=200, window=5, min_count=5, workers=8)
    model_skip_gram.save("./model_skip_gram.model")

    ### compute the similarity between frequently appeared words
    model_cbow = models.word2vec.Word2Vec.load("/data/dxy/codes/NLP_homework/NLP_homework4/model_cbow.model")
    model_skip_gram = models.word2vec.Word2Vec.load("/data/dxy/codes/NLP_homework/NLP_homework4/model_skip_gram.model")

    character_names = ["黄蓉", "杨过", "张无忌", "令狐冲", "韦小宝", "峨嵋派", "屠龙刀", "蛤蟆功", "葵花宝典"]
    print("Results of CBOW:")
    for tmp_word in character_names:
        print("Related words of {}: ".format(tmp_word), model_cbow.wv.most_similar(tmp_word, topn=5))
    print("------------------")
    print("Results of Skip Gram:")
    for tmp_word in character_names:
        print("Related words of {}: ".format(tmp_word), model_skip_gram.wv.most_similar(tmp_word, topn=5))

    ### get frequent words
    stop_words_list = []
    for tmp_file_name in os.listdir("/data/dxy/codes/NLP_homework/NLP_homework4/stopwords/"):      # replace this path with the stopwords path
        with open("/data/dxy/codes/NLP_homework/NLP_homework_2-main/stopwords/"+tmp_file_name, "r", encoding="utf-8", errors="ignore") as f:
            stop_words_list.extend([word.strip('\n') for word in f.readlines()])

    print("Getting the mostly frequent words in the corpus...")
    with open("./corpus.txt", "r", encoding="utf-8", errors="ignore") as tmp_file:
        whole_corpus = tmp_file.read()
        whole_corpus = whole_corpus.replace("\n", "")
        whole_corpus = whole_corpus.split(" ")
        words_counter = Counter(whole_corpus)

    frequent_words = []
    # most_frequent = sorted([(k, v) for (k, v) in words_counter.items()], key=lambda x: x[1], reverse=True)[:50]
    for tmp_key, tmp_value in words_counter.items():
        if tmp_value >= 50:
            if tmp_key not in stop_words_list:
                frequent_words.append(tmp_key)

    ### tSNE visualization
    print("tSNE visualization...")
    word_vectors = []
    for tmp_word in frequent_words:
        try:
            word_vectors.append(model_skip_gram.wv[tmp_word])
        except:
            continue

    tSNE = TSNE()
    word_embeddings = tSNE.fit_transform(word_vectors)
    classifier = KMeans(n_clusters=16)
    classifier.fit(word_embeddings)
    labels = classifier.labels_

    min_left = min(word_embeddings[:, 0])
    max_right = max(word_embeddings[:, 0])
    min_bottom = min(word_embeddings[:, 1])
    max_top = max(word_embeddings[:, 1])

    markers = ["bo", "go", "ro", "co", "mo", "yo", "ko", "bx", "gx", "rx", "cx", "mx", "yx", "kx", "b>", "g>"]

    for i in range(len(word_embeddings)):
        plt.plot(word_embeddings[i][0], word_embeddings[i][1], markers[labels[i]])
    plt.axis([min_left, max_right, min_bottom, max_top])
    plt.savefig("./tSNE.png")
