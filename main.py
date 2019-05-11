# -*- coding: utf-8 -*- 
import os
from datapreprocess import preprocess
import train_eval
import fire
from QA_data import QA_test
from config import Config



def chat(**kwargs):
    
    opt = Config()
    for k, v in kwargs.items(): #设置参数
        setattr(opt, k, v)   

    searcher, sos, eos, unknown, word2ix, ix2word = train_eval.test(opt)

    if os.path.isfile(opt.corpus_data_path) == False:
        preprocess()

    while(1):
        input_sentence = input('Doragd > ')
        if input_sentence == 'q' or input_sentence == 'quit' or input_sentence == 'exit': break
        if opt.use_QA_first:
            query_res = QA_test.match(input_sentence)
            if(query_res == tuple()):
                output_words = train_eval.output_answer(input_sentence, searcher, sos, eos, unknown, opt, word2ix, ix2word)
            else:
                output_words = "您是不是要找以下问题: " + query_res[1] + '，您可以尝试这样: ' + query_res[2]
        else:
            output_words = train_eval.output_answer(input_sentence, searcher, sos, eos, unknown, opt, word2ix, ix2word)
        print('BOT > ',output_words)

    QA_test.conn.close()
    
if __name__ == "__main__":
    fire.Fire()
