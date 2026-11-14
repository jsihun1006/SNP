import numpy as np
import pytest
from ..geometry import OpenCVCamera, SynLocCamera, project_ground


def down_camera(distortion=None, bounds=None):
    return OpenCVCamera(2000,1200,[[500,0,1000],[0,500,600],[0,0,1]],
                        np.diag([1.,-1.,-1.]),[0,0,10],distortion,bounds)


@pytest.mark.parametrize("dist", [None,[.08,-.01,.001,-.002,.001]])
def test_world_image_world_at_edges(dist):
    camera=down_camera(dist)
    for xyz in ([0,0,0],[15,8,0],[-15,-8,0],[1,-1,0]):
        uv=camera.world_to_pixel(xyz)
        world=camera.pixel_to_ground(uv)
        assert world.valid, world.reason
        assert np.allclose([world.x,world.y,world.z],xyz,atol=1e-7)


def test_far_world_point():
    r=[[1,0,0],[0,0,-1],[0,1,0]]
    camera=OpenCVCamera(2000,1200,[[500,0,1000],[0,500,600],[0,0,1]],r,[0,10,0])
    xyz=[400,1000,0]
    world=camera.pixel_to_ground(camera.world_to_pixel(xyz))
    assert world.valid and np.allclose([world.x,world.y],xyz[:2],atol=1e-7)
    assert camera.pixel_to_ground([1000,600]).reason == "parallel_ray"
    assert camera.pixel_to_ground([1000,500]).reason == "behind_camera"


def test_missing_camera_and_invalid_points():
    assert not project_ground(None,[100,100]).valid
    camera=down_camera(bounds=[-1,-1,1,1])
    assert camera.pixel_to_ground([1500,600]).reason == "outside_pitch"
    for uv in ([float('nan'),1],[1,float('inf')],[-1,5],[2000,5]):
        assert not camera.pixel_to_ground(uv).valid


def test_singular_calibration():
    with pytest.raises(ValueError):
        SynLocCamera(100,100,np.zeros((3,4)),[1,0],[1,0])


def test_synloc_equations_against_official_camera():
    import torch
    from sskit.camera import world_to_image, image_to_ground, normalize, unnormalize
    p=np.column_stack([np.diag([1.,-1.,-1.]),[0,0,10]])
    camera=SynLocCamera(3840,2160,p,[1,0],[1,0])
    for xyz in ([0.,0.,0.],[2.,1.,0.],[-2.,-1.,0.]):
        point=torch.tensor(xyz,dtype=torch.float64)
        official=unnormalize(world_to_image(torch.tensor(p),torch.tensor([1.,0.],dtype=torch.float64),point),(3,2160,3840))
        uv=camera.world_to_pixel(xyz)
        assert np.allclose(uv,official.numpy(),atol=1e-8)
        world=camera.pixel_to_ground(uv)
        official_xyz=image_to_ground(torch.tensor(p),torch.tensor([1.,0.],dtype=torch.float64),normalize(official,(3,2160,3840)))
        assert world.valid and np.allclose([world.x,world.y,world.z],official_xyz.numpy(),atol=1e-8)
