import spacy
from nltk import Tree

# import nltk as nlp


en_nlp = spacy.load('en')
doc = en_nlp(
    u"Bell, based in Los Angeles, makes and distributes electronic, computer and building products.")
# music here, apart from food being best, is good
print(doc)


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

sentences = list(doc.sents)

aspect = "products"
node_stack = []

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
                node_stack = []
                dep_root = child
                sent_list = []
                sent_list.append(y)
                # sent_list.append(dep_root.string.strip())

                for inn_child in dep_root.children:
                    if inn_child.string.strip() not in sent_list:
                        sent_list.append(inn_child.string.strip())

                while dep_root.head != dep_root:
                    dep_root = dep_root.head
                    sent_list.append(dep_root.string.strip())
                    for inn_child in dep_root.children:
                        if inn_child.string.strip() not in sent_list:
                            sent_list.append(inn_child.string.strip())
                print(sent_list)
                break

            node_stack.append(child)


def parse_tree():
    pass
