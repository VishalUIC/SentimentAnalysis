import spacy
from nltk import Tree
import re

valid_count = 0
invalid_count = 0


def concact_aspect_term(text, aspect_term):
    # print(text)
    # print(aspect_term)
    aspect_term = aspect_term.replace("(", "\(")
    aspect_term = aspect_term.replace(")", "\)")
    text = text.replace("[comma]", ",")
    try:
        found = re.search("" + aspect_term, text).group()
    except (AttributeError,) as e:
        # AAA, ZZZ not found in the original string
        found = ''  # apply your error handling
    replace = '_'.join(found.split())
    text_new = text.replace(found, replace)
    aspect_term_new = '_'.join(aspect_term.split())
    return text_new, aspect_term_new


# text, aspect_term = concact_aspect_term(
#   "while the keyboard itself is alright[comma] the plate around it is cheap plastic and makes a hollow sound when using the mouse command buttons.",
#  "mouse command buttons")


def generate_weightage_window(text, aspect_term):
    # chars_to_remove = ['.', '!', '?', '\\', '/', ':', '-', ';', '(', ')']
    # text_new = text.translate(None, ' '.join(chars_to_remove))
    # print(text)
    text_new = text.replace('/', ' ')
    # print(text_new)
    text_list = text_new.split()
    new_text_list = map(lambda it: it.strip(',.!:-$()[]{}\"\'\\;?/_'), text_list)
    # print(aspect_term)
    # print(text)

    try:
        aspect_term_index = new_text_list.index(aspect_term)
        global valid_count
        valid_count += 1
    except ValueError:
        global invalid_count
        invalid_count += 1
        return text
    for index, word in enumerate(text_list):
        if aspect_term_index == index:
            text_new += ' ' + word
            continue
        if abs(aspect_term_index - index) <= 4:
            counter = abs(abs(aspect_term_index - index) - 4) + 1
            while counter > 0:
                text_new += ' ' + word
                counter -= 1
    # print(text_new)
    return text_new


# generate_weightage_window(text, aspect_term)


# import nltk as nlp
def generate_dependency_wordbag(text, aspect_term):
    en_nlp = spacy.load('en')
    doc = en_nlp(text)
    # music here, apart from food being best, is good
    # print(doc)

    sentences = list(doc.sents)

    aspect = aspect_term
    node_stack = []
    sent_list = []

    for sentence in sentences:

        root_token = sentence.root
        x = root_token.children
        node_stack = []
        node_stack.append(root_token)
        while len(node_stack) != 0:
            cur_root_token = node_stack[-1]
            node_stack.pop()
            for child in cur_root_token.children:
                y = child.string.strip()
                if y == aspect:
                    dep_root = child.head
                    sent_list.append(y)
                    sent_list.append(dep_root.string)
                    for inn_child in dep_root.children:
                        if len(list(inn_child.children)) == 0 and inn_child.string is not y:
                            sent_list.append(inn_child.string)
                    # print(sent_list)
                node_stack.append(child)
    print(sent_list)
    return_text = ' '.join(str(e) for e in sent_list)
    print(return_text)
    return return_text


# generate_dependency_wordbag(u"The movie is good but the actors are bad", "movie")


def parse_tree():
    pass


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

# [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
