import json
import numpy as np
from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

def find_best_match(value, sentence): # value=ontology sentence=predicted sentence when inferencing
    bidx = 0
    best_score = 0
    min_length = min(len(value), len(sentence))
    for idx in range(len(sentence)-min_length+1):
        pres_score = len(np.where(np.array(list(value)[: min_length])==np.array(list(sentence)[idx: idx+min_length]))[0])
        (best_score, bidx) =  (pres_score, idx) if pres_score > best_score else (best_score, bidx)
    return bidx, best_score / len(value)

def cancel_parenthesis(sentence):
    new_sentence = []
    left_components = sentence.split(')')
    for comp in left_components:
        if '(' not in comp:
            new_sentence.append(comp)
        else:
            comp_list = comp.split('(')
            if len(comp_list) == 2:
                new_sentence.append(comp_list[0])
    return 'å'.join(new_sentence)

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt, restore_label=False)
                ex_restore = cls(utt, restore_label=True)
                examples.append(ex)
                examples.append(ex_restore)
        return examples

    @classmethod
    def analyze_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        wrong_label_num = 0
        total_label_num = 0
        for data in datas:
            for sentence in data:
                for label in sentence['semantic']:
                    bidx = sentence['asr_1best'].find(label[2])
                    total_label_num += 1
                    if bidx == -1:
                        wrong_label_num += 1
        print("wrong label rate:", wrong_label_num / total_label_num)

    def __init__(self, ex: dict, restore_label=False):
        super(Example, self).__init__()
        self.ex = ex

        self.utt = ex['asr_1best'] if restore_label == False else cancel_parenthesis(ex['manual_transcript'])
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx, score = find_best_match(value, self.utt)
            if score >= 0.5:
                actual_legth = min(len(value), len(self.utt)-bidx)
                self.tags[bidx: bidx + actual_legth] = [f'I-{slot}'] * actual_legth
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
