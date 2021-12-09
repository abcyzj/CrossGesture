from torch.utils.data import Dataset

from common.camera import camera_to_world


class MocapDataset(Dataset):
    def __init__(self, fps, skeleton, camera):
        super().__init__()

        self._skeleton = skeleton
        self._fps = fps
        self._camera = camera
        
    def __getitem__(self, _):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
        
    def fps(self):
        return self._fps
    
    def skeleton(self):
        return self._skeleton

    def camera(self):
        return self._camera

    def camera_to_world(self, poses):
        if 'orientation' not in self._camera:
            return poses
        rot = self._camera['orientation']
        return camera_to_world(poses, R=rot, t=0)
