import sentencepiece as spm

def train(input_file,vocab_size,model_name,model_type,character_coverage):
    input_argument = (
        '--input=%s '
        '--model_prefix=%s '
        '--model_size=%s '
        '--model_type=%s '
        '--character_coverage=%s '
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    )

    cmd = input_argument % (input_file,model_name,vocab_size,model_type,character_coverage)
    spm.SentencePieceTrainer.Train(cmd)

def run():
    en_input='../data/corpus.en'
    en_vocab_size=32000
    en_model_name='eng'
    en_model_type='bpe'
    en_character_coverage=1.0

    train(en_input,en_vocab_size,en_model_name,en_model_type,en_character_coverage)

    ch_input='../data/corpus.ch'
    ch_vocab_size=32000
    ch_model_name='ch'
    ch_model_type='bpe'
    ch_character_coverage=0.9995

    train(ch_input,ch_vocab_size,ch_model_name,ch_model_type,ch_character_coverage)

def test():
    sp = spm.SentencePieceProcessor()
    text='美国总统今天抵达夏威夷'

    sp.Load('./cnn.model')
    print(sp.EncodeAsPieces(text))

    print(sp.EncodeAsIds(text))

    a =[12907,277,7419,7318,10304,20724]
    print(sp.DecodeIds(a))

if __name__ == '__main__':
    run()
    # test()

