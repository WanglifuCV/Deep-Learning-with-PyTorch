# -*- coding:utf-8 -*-
# 有几部分想进行优化和修正：
# 1. 模型的训练部分的代码不明晰，想优化一下
# 2. 模型现在跑在CPU上，转到GPU上面去。

# Building and training a basic character-level RNN to classify words.

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os.path as osp
import torch
import random
import time
import math


def find_files(path): return glob.glob(path)

txt_files = 'NLP/data/names/*.txt'

print(find_files(txt_files))


import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII
def unicode_to_ascii(s):
    """
    将从文件读取的unicode编码的数据转换为ascii编码的数据
    """
    output = ''.join(
        [
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != 'Mn' and c in all_letters
        ]
    )
    return output


category_lines = {}
all_categories = []

def read_lines(file_name):
    """
    读取文件
    """
    with open(file_name, encoding='utf-8') as file_reader:
        lines = file_reader.readlines()
        return [unicode_to_ascii(line.strip()) for line in lines]

for file_name in find_files(txt_files):
    category = osp.splitext(osp.basename(file_name))[0]
    all_categories.append(category)
    lines = read_lines(file_name)
    category_lines[category] = lines

n_categories = len(all_categories)
print(category_lines)
print(n_categories)


def letter_to_index(letter):
    return all_letters.find(letter)


def letter_to_tensor(letter):
    """
    Just for demonstration, turn a letter into a [1 x n_letters] Tensor
    """
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter=letter)] = 1.0
    return tensor

def line_to_tensor(line):
    """
    Turn a line into a [line_length x 1 x n_letter]
    """

    tensor = torch.zeros(len(line), 1, n_letters)

    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter=letter)] = 1.0
    return tensor

print(letter_to_tensor('J'))
print(line_to_tensor('Jones').size())

class RNN(torch.nn.Module):
    """
    这是直接实现了一个简单RNN
    """
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)

        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

criterion = torch.nn.NLLLoss()

input = line_to_tensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)


def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(category_from_output(output))


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def train(category_tensor, line_tensor, learning_rate=0.005):

    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


n_iters = 100000
print_every = 100
plot_every = 10

current_loss = 0
all_losses = []


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{}m {}s'.format(m, s)

start = time.time()


print('Start training')
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor=category_tensor, line_tensor=line_tensor)

    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ {}'.format(category)
        print(
            'Iter: {} Iter-percent: {} Time:{} Loss: {} Line: {} Guess: {} Correct: {}'.format(iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct)
        )
    
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)

        current_loss = 0