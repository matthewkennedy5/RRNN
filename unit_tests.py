import pdb
import trainer
import structure_utils
import tree_methods

not_gru = [['x', 'h', 0, 'mul', 'tanh', 'G0'],
           ['h', 'G0', 2, 'add', 'tanh', 'G1'],
           ['x', 'G1', 3, 'mul', 'tanh', 'G2'],
           ['x', 'h', 0, 'mul', 'tanh', 'G3'],
           ['h', 'G3', 2, 'add', 'tanh', 'G4'],
           ['x', 'G4', 3, 'mul', 'tanh', 'G5'],
           ['G2', 'G5', 3, 'mul', 'tanh', 'G6'],
           ['h', 'G6', 3, 'mul', 'tanh', 'G7'],
           ['h', 'G7', 3, 'mul', 'tanh', 'G8']]

not_gru2 = [['h', 'x', 0, 'add', 'sigmoid', 'z1'],
            ['h', 'x', 0, 'add', 'sigmoid', 'r'],
            ['h', 'x', 0, 'add', 'tanh', 'z2'],
            ['0', 'z2', 0, 'add', 'minus', '1-z'],
            ['h', 'z1', 0, 'mul', 'identity', 'z*h'],
            ['r', 'h', 0, 'mul', 'identity', 'r*h'],
            ['x', 'r*h', 0, 'add', 'tanh', 'h_tilde'],
            ['1-z', 'h_tilde', 0, 'mul', 'identity', '(1-z)*h_tilde'],
            ['z*h', '(1-z)*h_tilde', 0, 'add', 'identity', 'ht']]

# k values are wrong right now
gru1 = [['x', 'h', 0, 'add', 'sigmoid', 'z1'],
        ['x', 'h', 1, 'add', 'sigmoid', 'r'],
        ['x', 'h', 1, 'add', 'sigmoid', 'z2'],
        ['r', 'h', 2, 'mul', 'identity', 'r*h'],
        ['x', 'r*h', 3, 'add', 'tanh', 'h_tilde'],
        ['z1', '0', 2, 'add', 'minus', '1-z'],
        ['1-z', 'h_tilde', 3, 'mul', 'identity', '(1-z)*h_tilde'],
        ['z2', 'h', 3, 'mul', 'identity', 'z*h'],
        ['z*h', '(1-z)*h_tilde', 3, 'add', 'identity', 'ht']]

gru2 = [['h', 'x', 0, 'add', 'sigmoid', 'z1'],
        ['x', 'h', 1, 'add', 'sigmoid', 'r'],
        ['x', 'h', 1, 'add', 'sigmoid', 'z2'],
        ['r', 'h', 2, 'mul', 'identity', 'r*h'],
        ['x', 'r*h', 3, 'add', 'tanh', 'h_tilde'],
        ['0', 'z1', 2, 'add', 'minus', '1-z'],
        ['1-z', 'h_tilde', 3, 'mul', 'identity', '(1-z)*h_tilde'],
        ['z2', 'h', 3, 'mul', 'identity', 'z*h'],
        ['z*h', '(1-z)*h_tilde', 3, 'add', 'identity', 'ht']]

gru3 = [['h', 'x', 0, 'add', 'sigmoid', 'z1'],
        ['0', 'z1', 2, 'add', 'minus', '1-z'],
        ['h', 'x', 1, 'add', 'sigmoid', 'r'],
        ['h', 'r', 2, 'mul', 'identity', 'r*h'],
        ['x', 'r*h', 3, 'add', 'tanh', 'h_tilde'],
        ['1-z', 'h_tilde', 3, 'mul', 'identity', '(1-z)*h_tilde'],
        ['x', 'h', 1, 'add', 'sigmoid', 'z2'],
        ['h', 'z2', 3, 'mul', 'identity', 'z*h'],
        ['(1-z)*h_tilde', 'z*h', 3, 'add', 'identity', 'ht']]

gru4 = [['h', 'x', 0, 'add', 'sigmoid', 'z1'],
        ['h', 'x', 0, 'add', 'sigmoid', 'r'],
        ['h', 'x', 0, 'add', 'sigmoid', 'z2'],
        ['0', 'z2', 0, 'add', 'minus', '1-z'],
        ['h', 'z1', 0, 'mul', 'identity', 'z*h'],
        ['r', 'h', 0, 'mul', 'identity', 'r*h'],
        ['x', 'r*h', 0, 'add', 'tanh', 'h_tilde'],
        ['1-z', 'h_tilde', 0, 'mul', 'identity', '(1-z)*h_tilde'],
        ['z*h', '(1-z)*h_tilde', 0, 'add', 'identity', 'ht']]

def traverse(tree):
    if tree is not None:
        print(tree.name)
        traverse(tree.left)
        traverse(tree.right)
    else:
        print()

# Tests for structures_are_equal
# assert(not structure_utils.structures_are_equal(not_gru, gru1))
# assert(structure_utils.structures_are_equal(gru1, gru2))
# assert(structure_utils.structures_are_equal(gru2, gru1))
# # traverse(structure_utils.structure2tree(gru1))
# # traverse(structure_utils.structure2tree(gru3))
# assert(structure_utils.structures_are_equal(gru1, gru3))
# assert(structure_utils.structures_are_equal(gru3, gru1))
# assert(structure_utils.structures_are_equal(gru3, gru2))
# assert(structure_utils.structures_are_equal(gru2, gru3))
# assert(structure_utils.structures_are_equal(gru4, gru1))
# assert(structure_utils.structures_are_equal(gru2, gru4))
# assert(not structure_utils.structures_are_equal(gru2, not_gru2))


################### Tests for label (tree_methods) ########################

def print_tree(tree):
    try:
        print(tree.number)
    except AttributeError:
        print('Leaf')
    if tree.leftchild is not None:
        print_tree(tree.leftchild)
    if tree.rightchild is not None:
        print_tree(tree.rightchild)

# Tree 1
vector = 0
node1 = tree_methods.Node(vector)
node2 = tree_methods.Node(vector)
leaf1 = tree_methods.Node(vector)
leaf2 = tree_methods.Node(vector)

node1.leftchild = leaf1
node1.rightchild = node2
node2.leftchild = leaf2

result = tree_methods.label(node1)
print_tree(result)

print()
print()

# Tree 2
node1 = tree_methods.Node(vector)
node2 = tree_methods.Node(vector)
node3 = tree_methods.Node(vector)
leaf4 = tree_methods.Node(vector)
leaf5 = tree_methods.Node(vector)
node1.rightchild = node2
node2.rightchild = node3
node3.leftchild = leaf4
node3.rightchild = leaf5
tree_methods.label(node1)
print_tree(node1)
print('\n')

# Tree 3
node1 = tree_methods.Node(vector)
node2 = tree_methods.Node(vector)
node3 = tree_methods.Node(vector)
leaf4 = tree_methods.Node(vector)
leaf5 = tree_methods.Node(vector)
leaf6 = tree_methods.Node(vector)
leaf7 = tree_methods.Node(vector)
node1.leftchild = node2
node1.rightchild = node3
node2.leftchild = leaf4
node2.rightchild = leaf5
node3.leftchild = leaf6
node3.rightchild = leaf7
tree_methods.label(node1)
print_tree(node1)

