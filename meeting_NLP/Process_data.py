#import start_token, end_token
import collections
import  numpy as np
def process_poems(file_name): # 每一行都是一首诗
    poems = []
    with open(file_name,'r',encoding='utf-8') as f:
        for line in f.readlines():
            title,content = line.strip().split(':')
            if ' ' in content or '{' in content:
                continue
            if len(content) < 5 or len(content>) > 1000:
                continue
       #     content = start_token + end_token
            poems.append(content)
    poems = sorted(poems, key= lambda 1:len(line) )
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words) # 进行字统计
    count_paris = sorted(counter.items(),key =lambda x:x[-1]) # 词和个数的组合
    words, _ = zip(*count_paris)
    word_int_mapping= dict(zip(words,range(len(words))))

    poems_vecor = [list(map(lambda word:word_int_mapping.get(word,len(words))))]

    return poems_vecor,word_int_mapping,words

def generate_batch(batch_size,poems_vector,word_to_int):
    n_chunk =  len(poems_vector) / batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i *batch_size
        end_index =  start_index + batch_size
        """
        数据需要补全，paddin，用空格键补齐
        pytorch 和 dynamic_rnn的区别
        """
        batches = poems_vector[start_index:end_index]
        length = max(map(len,batches))
        x_data = np.full((batch_size,length),word_to_int(' '),np.int32)
        for row in range(batch_size):
            x_data[row,:len(batches[row])] = batches[row]

        """
        y值的定义是抽象的，但是跟我想的差不多
        """
        y_data = np.copy(x_data)
        y_data[:,:-1] = x_data[:,1:]
        x_batches.append((x_data))
        y_batches.append((y_data))
    return x_batches,y_batches

