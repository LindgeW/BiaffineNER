class Instance(object):
    def __init__(self, word: str,
                 pos_tag: str,
                 ner_tag: str):
        self.word = word          # 词
        self.pos_tag = pos_tag    # 词性
        self.ner_tag = ner_tag    # ner标签

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


