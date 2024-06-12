from abc import ABC, abstractmethod
from .node import Node
from .icons import Icons

class StyleNode: 
    def __init__(self, root: Node, icons: Icons) -> None:
        self._root = root
        self._icons = icons

    @abstractmethod
    def render(self) -> None:
        pass

class StyleNodeFactory(ABC): # 抽象工厂类
    @abstractmethod
    def create(self, root: Node, icons: Icons) -> StyleNode: #工厂方法
        pass
