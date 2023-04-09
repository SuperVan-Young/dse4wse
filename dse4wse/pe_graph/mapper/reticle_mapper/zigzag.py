from .base import BaseReticleMapper

class ZigZagReticleMapper(BaseReticleMapper):
    def __init__(self,
                 reticle_array_height: int,
                 reticle_array_width: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.reticle_array_height = reticle_array_height
        self.reticle_array_width = reticle_array_width
        self._curve = self.__init_curve()

    def __init_curve(self):
        mapping_list = []
        for i in range(self.reticle_array_height):
            j_range = range(0, self.reticle_array_width) if (i % 2 == 0) else range(self.reticle_array_width - 1, -1, -1)
            mapping_list += [(i, j) for j in j_range]
        return mapping_list

    def __call__(self, virtual_reticle_id: int):
        assert virtual_reticle_id >= 0 and virtual_reticle_id < len(self._curve)
        return self._curve[virtual_reticle_id]