from .base import BaseReticleMapper

class XYReticleMapper(BaseReticleMapper):
    def __init__(self,
                 reticle_array_height: int,
                 reticle_array_width: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.reticle_array_height = reticle_array_height
        self.reticle_array_width = reticle_array_width

    def __call__(self, virtual_reticle_id: int):
        x = virtual_reticle_id % self.reticle_array_height
        y = virtual_reticle_id // self.reticle_array_height
        assert y >= 0 and y < self.reticle_array_width
        return (x, y)