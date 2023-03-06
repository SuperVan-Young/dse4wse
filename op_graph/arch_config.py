
from typing import Dict
from functools import reduce

class ArchConfig():

    def __init__(self, config: Dict) -> None:
        self.config = config

    def __repr__(self) -> str:
        main_str = "ArchConfig {\n"
        for name, val in self.config.items():
            main_str += f"  {name}: {val}\n"
        main_str += "}\n"
        return main_str
    
    def _shallow_repr(self) -> str:
        def get_brief_name(name: str) -> str:
            brief = name.split("_")
            brief = [s[0] for s in brief]
            brief = reduce(lambda x, y: x+y, brief)
            return brief
        
        brief_config = [f"{get_brief_name(name)}{val}" for name, val in self.config.items()]
        brief_config = "_".join(brief_config)
        return brief_config