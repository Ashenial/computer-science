import json
from .TreeStyle import TreeStyleNodeFactory
from .RectStyle import RectStyleNodeFactory
from .node import NodeFactory
from .icons import Icons

class StyleBuilder:
    def __init__(self):
        self._styles_factory = {
            'tree': TreeStyleNodeFactory(),
            'rect': RectStyleNodeFactory()
        }
        self._icons = {
            'void': Icons(' ', ' '),
            'pokerface': Icons('\u2666', '\u2664'), # ♢ ♤
            'chess': Icons('\u2656', '\u2659'), # ♖ ♙
        }

    def create_style(self, file: str, icons: str, style: str):
        icon = self._icons[icons]
        style_factory = self._styles_factory[style]

        json_node = NodeFactory(file).create()
        return style_factory.create(json_node, icon)
    
    def load_icons(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            icons_dict = json.load(f)

        for name, icons in icons_dict.items():
            self._icons[name] = Icons(icons['composite'], icons['leaf'])


