import json

if __name__ == '__main__':
    files = ['train','dev','test']
    ch_path = 'corpus.ch'
    en_path = 'corpus.en'
    ch_lines =[]
    en_lines =[]

    for file in files:
        corpus = json.load(open('.json/'+file+'.json','r',encoding='utf-8'))
        for item in corpus:
            en_lines.append(item[0]+'\n')
            ch_lines.append(item[1]+'\n')

    with open(ch_path,'w',encoding='utf-8') as f:
        f.writelines(ch_lines)

    with open(en_path,'w',encoding='utf-8') as f:
        f.writelines(en_lines)

    # lines of Chinese: 252,777
    print("lines of Chinese: ", len(ch_lines))
    # 输出英文句子的行数
    # lines of English: 252,777
    print("lines of English: ", len(en_lines))
    # 输出完成提示信息
    print("-------- Get Corpus ! --------")