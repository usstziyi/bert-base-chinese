"""从 display 表直接读取按展示顺序排列的 words 列表。"""
from collections import Counter
from itertools import groupby
from pathlib import Path
import re

import openpyxl

ChineseEEG_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'ChineseEEG'
SEG_NOVEL_PATH = ChineseEEG_ROOT / 'derivatives' / 'novels' / 'segmented_novel'


def _load_display_words(path, run_num):
    """每条 display 记录生成一个字符字典，包括章节展示记录。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        records = wb.active.iter_rows(values_only=True)
        headers = next(records, ())
        required = ('Chinese_text', 'index', 'main_row', 'row_num')
        if not set(required).issubset(headers):
            raise ValueError(f'{path}: display table must contain {required}')

        words = []
        row_num, char_idx = 0, 0
        previous_text, previous_index = None, -1
        for display_row, values in enumerate(records, start=2):
            if all(value is None for value in values):
                continue
            record = dict(zip(headers, values))
            try:
                if record['Chinese_text'] is None:
                    raise ValueError('Missing displayed text')
                index = int(record['index']) # 高亮字符索引
                main_row = int(record['main_row'])
                row_count = int(record['row_num'])
                if any(record[k] != value for k, value in
                       zip(('index', 'main_row', 'row_num'), (index, main_row, row_count))):
                    raise ValueError('Display coordinates must be integers')
                lines = [line for line in str(record['Chinese_text']).split('\n') if line]
                if row_count not in (1, 2, 3) or not 0 <= main_row < len(lines):
                    raise ValueError('Invalid display row coordinates')
                text = lines[main_row] # 截取中间高亮字符所在行文本
                if not 0 <= index < len(text):
                    raise ValueError('Highlight index outside displayed row')
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{path}, row {display_row}: invalid display record ({exc})') from exc

            # 索引重新开始时，相邻的相同文本也计为新的一行。
            if text != previous_text or index <= previous_index: # 进入新的文本行时，递增行号，并重置行内高亮序号
                row_num += 1 # 当前行号从 1 开始计数
                char_idx = 0 # 当前行内高亮序号从 0 开开始计数
            words.append(dict(
                text=text[index], 
                run_num=run_num, # run 编号，从 1 开始
                row_num=row_num, # 当前行号，从 1 开始
                char_idx=char_idx,
                is_chapter=row_count == 1 and text.isdecimal(), 
            ))
            previous_text, previous_index = text, index
            char_idx += 1
        if not words:
            raise ValueError(f'No display records in {path}')
        for _, row_words in groupby(words, key=lambda word: word['row_num']):
            row_words = list(row_words)
            for word in row_words:
                word['n_chars'] = len(row_words)
        return words
    finally:
        wb.close()


def load_text(novel_name='LittlePrince', run_num=7):
    """返回扁平 words 列表：words[i] 就是一条 display 记录。

    text 为 index 指向的字符，保留汉字、字母、数字及高亮符号。
    run_num/row_num 从 1 开始，char_idx 从 0 开始。
    is_chapter 标识章节展示记录。
    n_chars 是该行展示记录数。run_num=None 自动读取全部连续编号的 run。
    """
    folder = SEG_NOVEL_PATH / novel_name
    if run_num is None:
        pattern = re.compile(r'segmented_Chinense_novel_run_(\d+)_display\.xlsx')
        numbers = sorted(int(m[1]) for p in folder.glob('*.xlsx') if (m := pattern.fullmatch(p.name)))
        if not numbers:
            raise FileNotFoundError(f'No display runs found in {folder}')
        if numbers != list(range(1, numbers[-1] + 1)):
            raise ValueError(f'Display run numbers must be consecutive from 1: {numbers}')
        run_num = numbers[-1]
    if isinstance(run_num, bool) or not isinstance(run_num, int) or run_num <= 0:
        raise ValueError('run_num must be a positive integer or None')

    
    words = []
    for run in range(1, run_num + 1):
        path = folder / f'segmented_Chinense_novel_run_{run}_display.xlsx'
        words.extend(_load_display_words(path, run))
    return words



if __name__ == '__main__':
     words = load_text()
     for word in words[:100]:
        print(word)

        

# if __name__ == '__main__':
#     words = load_text()
#     print(f'total: {len(words)}')
#     run_counts = Counter(word['run_num'] for word in words)
#     for run_num in sorted(run_counts):
#         print(f'run {run_num}: {run_counts[run_num]} words')

#     chapter, chapter_counts = 0, Counter()
#     for word in words:
#         if word['is_chapter']:
#             chapter += 1
#         chapter_counts[chapter] += 1
#     for chapter_num in sorted(chapter_counts):
#         label = 'front matter' if chapter_num == 0 else f'chapter {chapter_num}'
#         print(f'{label}: {chapter_counts[chapter_num]} words')
