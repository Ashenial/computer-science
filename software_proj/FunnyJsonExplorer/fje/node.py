from abc import ABC, abstractmethod
from typing import List, Callable, Union
import json

id = 0

class Node(ABC):
    def __init__(self, name: str, level: int):
        global id 
        self._name = name
        self._level = level
        self._id = id
        id += 1

    @abstractmethod
    def is_leaf(self) -> bool:
        pass

    @abstractmethod
    def iterate(self, func: Callable[['Node'], None]):
        pass

    def is_root(self) -> bool:
        return self._level == 0

    def get_name(self) -> str:
        return self._name
    
    def get_id(self) -> int:
        return self._id
    
    def get_level(self) -> int:
        return self._level
    
class Composite(Node):
    def __init__(self, name, level):
        super().__init__(name, level)
        self.children: List[Node] = []
    
    def is_leaf(self) -> bool:
        return False

    def iterate(self, func: Callable[['Node'], None]):
        func(self)
        for child in self.children:
            child.iterate(func)

    def add_child(self, child: Node):
        self.children.append(child)
    
    def get_children(self) -> List[Node]:
        return self.children

    def __iter__(self):
        return iter(self.children)
    
class Leaf(Node):
    def __init__(self, name, level, value: Union[str, None]):
        super().__init__(name, level)
        self._value = value

    def is_leaf(self) -> bool:
        return True

    def iterate(self, func: Callable[['Node'], None]):
        func(self)

    def get_value(self) -> Union[str, None]:
        return self._value
    
class NodeFactory:
    def __init__(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.json_data = json.load(f)
    
    def create(self) -> Node:
        return self._create('', 0, self.json_data)

    def _create(self, name: str, level: int, obj) -> Node:
        if isinstance(obj, list):
            return self._create_composite_from_list(name, level, obj)
        elif isinstance(obj, dict):
            return self._create_composite_from_dict(name, level, obj)
        else:
            return self._create_leaf(name, level, obj)

    def _create_composite_from_list(self, name: str, level: int, obj) -> Composite:
        composite = Composite(name, level)
        for idx, item in enumerate(obj):
            child = self._create(f'Array[{idx}]', level + 1, item)
            composite.add_child(child)
        return composite

    def _create_composite_from_dict(self, name: str, level: int, obj) -> Composite:
        composite = Composite(name, level)
        for key, value in obj.items():
            child = self._create(key, level + 1, value)
            composite.add_child(child)
        return composite
    
    def _create_leaf(self, name: str, level: int, obj) -> Leaf:
        if obj is None:
            return Leaf(name, level, None)
        else:
            return Leaf(name, level, str(obj))