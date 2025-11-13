import sentencepiece as spm

def chinese_tokenizer_load():
    sp_chn = spm.SentencePieceProcessor()
    sp_chn.Load('{}.model'.format("./tokenizer/chn"))
    return sp_chn

def english_tokenizer_load():
    sp_en = spm.SentencePieceProcessor()
    sp_en.Load('{}.model'.format("./tokenizer/en"))
    return sp_en

