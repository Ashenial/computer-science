from .style import StyleNode, StyleNodeFactory
from .icons import Icons
from .node import Node, Leaf, Composite
#### Tree Style #####

class TreeStyleNode(StyleNode): 

    def __init__(self, root: Node, icons: Icons):
        super().__init__(root, icons)

    def plot(self) -> None:
        self._plot_branch('', '', self._root)

    def _plot(self, prefix: str, postfix: str, node: Node) -> None:
        if node.is_leaf():
            self._plot_leaf(prefix, node)
        else:
            self._plot_branch(prefix, postfix, node)
    
    def _plot_leaf(self, prefix: str, node: Leaf) -> None:
        value = node.get_value()
        if value is None:
            print(f'{prefix}{self._icons.leaf_icon} {node.get_name()}')
        else:
            print(f'{prefix}{self._icons.leaf_icon} {node.get_name()}: {value}')

    def _plot_branch(self, prefix_first: str, prefix_follow: str, node: Composite) -> None:
        if not node.is_root():
            print(f'{prefix_first}{self._icons.composite_icon} {node.get_name()}')
        children = node.get_children()
        if len(children) == 0:
            return
        for child in children[:-1]:
            self._plot(f'{prefix_follow} ├─', f'{prefix_follow} │  ',child)
        self._plot(f'{prefix_follow} └─', f'{prefix_follow}    ', children[-1])


class TreeStyleNodeFactory(StyleNodeFactory):
    def create(self, root: Node, icons: Icons) -> StyleNode:
        return TreeStyleNode(root, icons)