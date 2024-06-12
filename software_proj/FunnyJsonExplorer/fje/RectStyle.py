from .style import StyleNode, StyleNodeFactory
from .icons import Icons
from .node import Node
#### Rectangle Style #####

class RectangleStyleNode(StyleNode):

    def __init__(self, root: Node, icons: Icons):
        super().__init__(root, icons)
        self._grid_width = 16
        self._root.iterate(lambda node: self._update_grid_width(node))
        self.fl_detector = FirstLastDetector(root)
        
    def _update_grid_width(self, node: Node) -> None:
        prefix_length = max((node.get_level() - 1) * 3 + 2, 0)
        name_length = len(node.get_name()) + 2
        if node.is_leaf() and node.get_value() is not None:
            name_length += len(node.get_value()) + 2
        self._grid_width = max(self._grid_width, prefix_length + name_length + 2)
    
    def render(self) -> None:
        self._root.iterate(lambda node: self._render(node))

    def _render(self, node: Node) -> None:
        if node.is_root():
            return
        result = ''
        # first layer
        if self.fl_detector.is_first(node):
            result += "┌─"
        elif self.fl_detector.is_last(node):
            result += "└─"
        elif node.get_level() == 1:
            result += "├─"
        else:
            result += "│ "
        # straight lines
        if node.get_level() > 2:
            if self.fl_detector.is_last(node):
                result += '─┴─' * (node.get_level() - 2)
            else:
                result += ' │ ' * (node.get_level() - 2)
        # header
        if self.fl_detector.is_last(node):
            result += '─┴─'
        elif node.get_level() > 1:
            result += ' ├─'
        # icon
        if node.is_leaf():
            result += self._icons.leaf_icon + ' '
        else:
            result += self._icons.composite_icon + ' '
        # name
        result += node.get_name()
        # value
        if node.is_leaf() and node.get_value() is not None:
            result += f': {node.get_value()}'
        # padding
        result = f'{result} '.ljust(self._grid_width - 1, '─')
        # last layer
        if self.fl_detector.is_first(node):
            result += '─┐'
        elif self.fl_detector.is_last(node):
            result += '─┘'
        else:
            result += '─┤'
        print(result)


class RectStyleNodeFactory(StyleNodeFactory):

    def create(self, root: Node, icons: Icons) -> StyleNode:
        return RectangleStyleNode(root, icons)
    

class FirstLastDetector:

    def __init__(self, root: Node):
        self.first_visited = False
        root.iterate(lambda node: self.fn(node))

    def fn(self, node: Node):
        if node.is_root():
            return
        if not self.first_visited:
            self.first_visited = True
            self.first_id = node.get_id()
        else:
            self.last_id = node.get_id()

    def is_first(self, node: Node) -> bool:
        return node.get_id() == self.first_id
    
    def is_last(self, node: Node) -> bool:
        return node.get_id() == self.last_id