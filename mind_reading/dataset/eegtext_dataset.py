"""提供每行高亮数量给 EEG loader，并按字符位置配对。"""
from itertools import groupby

import torch
from torch.utils.data import Dataset

if __package__:
    from .load_eeg import SAMPLES_PER_CHAR, load_eeg
    from .load_text import load_text
else:
    from load_eeg import SAMPLES_PER_CHAR, load_eeg
    from load_text import load_text


class ChineseEEGDataset(Dataset):
    """include_padding 默认保留补零样本；相同文本共享 word_id。

    word_id 表示字词身份，word_idx 表示一次具体出现，不能用作同字对齐标签。
    ID 在整个 Dataset 内一致，Subset、shuffle 和 padding 过滤不会重新编号。
    """

    def __init__(self, novel_name='LittlePrince', filtered='filtered_0.5_30',
                 subject='sub-04', run_num=7, dtype=torch.float32, *,
                 include_chapters=False, include_padding=True):
        super().__init__()
        if not dtype.is_floating_point:
            raise ValueError('EEG dtype must be floating point')
        self.novel_name, self.filtered, self.subject = novel_name, filtered, subject
        self.dtype, self.include_chapters = dtype, include_chapters
        self.include_padding = include_padding
        self.words = load_text(novel_name, run_num)
        if not self.words:
            raise ValueError('No display characters found in the selected runs')
        self.word_to_id = {text: index for index, text in
                           enumerate(dict.fromkeys(word['text'] for word in self.words))}
        self.run_num = self.words[-1]['run_num']
        row_key = lambda word: (word['run_num'], word['row_num'])
        text_rows = [(key, list(items)) for key, items in groupby(self.words, key=row_key)]
        for key, row_words in text_rows:
            if ([word['char_idx'] for word in row_words] != list(range(len(row_words))) or
                    any(word['n_chars'] != len(row_words) or
                        word['is_chapter'] != row_words[0]['is_chapter'] for word in row_words)):
                raise ValueError(f'{key}: incomplete or unordered display characters')


        self.eeg_words = load_eeg(
            novel_name, filtered, subject, self.run_num,
            row_char_counts={key: len(items) for key, items in text_rows},
        )
        if len(self.words) != len(self.eeg_words):
            raise ValueError('EEG/text character counts differ')
        self.samples_per_char = SAMPLES_PER_CHAR
        self.samples = []
        for word_idx, (word, eeg_word) in enumerate(zip(self.words, self.eeg_words)):
            if any(word[key] != eeg_word[key] for key in ('run_num', 'row_num', 'char_idx')):
                raise ValueError(f'Word {word_idx}: EEG/text metadata differs')
            eeg_word.update(is_chapter=word['is_chapter'], n_chars=word['n_chars'])
            if word['is_chapter'] and not include_chapters:
                continue
            if eeg_word['is_padding'] and not include_padding:
                continue
            self.samples.append(
                dict(
                    word_idx=word_idx,
                    run_idx=word['run_num'] - 1,
                    row_num=word['row_num'], 
                    char_idx=word['char_idx']
                )
            )
        if not self.samples:
            raise ValueError('No display characters remain after chapter/padding filtering')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        word_idx = self.samples[index]['word_idx']
        word = self.words[word_idx]
        eeg_word = self.eeg_words[word_idx]
        return {
            **word, 
            **eeg_word,
            'eeg': torch.as_tensor(eeg_word['eeg'], dtype=self.dtype),
            'word_id': self.word_to_id[word['text']],
            'run_idx': word['run_num'] - 1, 
            'row_idx': word['row_num'] - 1,
        }


if __name__ == '__main__':
    dataset = ChineseEEGDataset(include_padding=True)
    print(len(dataset))

    dataset = ChineseEEGDataset(include_padding=False)
    print(len(dataset))
    for key,value in dataset[0].items():
        if key != 'eeg':
            print(f'{key}: {value}')
        else:
            print(f'{key}: {value.shape}')
