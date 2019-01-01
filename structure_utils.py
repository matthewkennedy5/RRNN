
LEFT_NAME = 0
RIGHT_NAME = 1
K = 2
BINARY_FUNC = 3
ACTIVATION = 4
NAME = 5

# k values are wrong right now
GRU_STRUCTURE = [['x', 'h', 0, 'add', 'sigmoid', 'z1'],
                 ['x', 'h', 1, 'add', 'sigmoid', 'r'],
                 ['x', 'h', 1, 'add', 'sigmoid', 'z2'],
                 ['r', 'h', 2, 'mul', 'identity', 'r*h'],
                 ['x', 'r*h', 3, 'add', 'tanh', 'h_tilde'],
                 ['z1', '0', 2, 'add', 'minus', '1-z'],
                 ['1-z', 'h_tilde', 3, 'mul', 'identity', '(1-z)*h_tilde'],
                 ['z2', 'h', 3, 'mul', 'identity', 'z*h'],
                 ['z*h', '(1-z)*h_tilde', 3, 'add', 'identity', 'ht']]


class TreeNode:
    """Simple class to store nodes of a structure tree.

    A structure tree just describes the structure of the update equations, and
    doesn't record vector information.
    """

    def __init__(self):
        self.left = None
        self.right = None
        self.k = None
        self.binary_func = None
        self.activation = None
        self.name = None

    def __eq__(self, tree2):
        """Returns True if the two trees have the same basic attributes.

        These attributes are children, name, binary function, and activation
        function.

        Input:
            tree2 - The tree to compare equality with
        """
        return (self.left == tree2.left and self.right == tree2.right and
                    self.name == tree2.name and self.binary_func == tree2.binary_func and
                        self.activation == tree2.activation)

    def is_leaf(self):
        return self.left is None and self.right is None


def leaf_nodes():
    """Returns a dict of the basic leaf vectors that can be used by the equations.

    Here they are x, h, and the zero vector.
    """
    x = TreeNode()
    x.name = 'x'
    h = TreeNode()
    h.name = 'h'
    zero = TreeNode()
    zero.name = '0'
    return {'x': x, 'h': h, '0': zero}


def structure2tree(structure):
    """Returns the tree representation of the given structure.

    Input:
        structure - List of nodes, where each node is a list of
            [leftname, rightname, k, binary_func, activation, name]. Child nodes
            must appear before their parents.

    Returns:
        node - TreeNode of the root of the structure tree.
    """
    nodes = leaf_nodes()
    for n in structure:
        node = TreeNode()
        if n[LEFT_NAME] not in nodes or n[RIGHT_NAME] not in nodes:
            print(n[LEFT_NAME])
            print(n[RIGHT_NAME])
            raise ValueError('Children must appear before their parents in \
                              the structure')
        node.left = nodes[n[LEFT_NAME]]
        node.right = nodes[n[RIGHT_NAME]]
        node.k = n[K]
        node.binary_func = n[BINARY_FUNC]
        node.activation = n[ACTIVATION]
        node.name = n[NAME]
        nodes[node.name] = node
    return node


def trees_are_isomorphic(tree1, tree2):
    """Returns True if the two trees are isomorphic."""
    if tree1 is None:
        return tree2 is None
    if tree2 is None:
        return tree1 is None
    # Check basic properties of the current node
    if tree1.binary_func != tree2.binary_func or tree1.activation != tree2.activation:
        return False
    # Check if they're the same leaf node
    if (tree1.is_leaf() or tree2.is_leaf()) and tree1 != tree2:
        return False
    # Recursive case
    return ((trees_are_isomorphic(tree1.left, tree2.left) and
                trees_are_isomorphic(tree1.right, tree2.right)) or
            (trees_are_isomorphic(tree1.left, tree2.right) and
                trees_are_isomorphic(tree1.right, tree2.left)))


def structures_are_equal(structure1, structure2):
    """Returns true if the two tree structures are isomorphic.

    If the two tree structures are isomorphic, that means they represent the
    same underlying equations and for all practical purposes are the same.

    Inputs:
        structure1 - A tree structure
        structure2 - A tree structure
    """
    # TODO: Do different permutations of Lk and Rk count as the same tree?
    if len(structure1) != len(structure2):
        return False

    tree1 = structure2tree(structure1)
    tree2 = structure2tree(structure2)
    return trees_are_isomorphic(tree1, tree2)
