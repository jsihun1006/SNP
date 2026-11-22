import pytest
from ..tiling.baat import Tile, expand_tile, adaptive_tiles, regular_grid


@pytest.mark.parametrize("box", [(90,20,130,60), (90,90,130,130)])
def test_edge_and_corner(box):
    t = expand_tile(Tile(0,0,100,100), [box], 300,300)
    assert t.x1 <= box[0] and t.y1 <= box[1] and t.x2 >= box[2]+5 and t.y2 >= box[3]+5
    assert t.converged


def test_chain_needs_multiple_iterations():
    t = expand_tile(Tile(0,0,100,100), [(90,10,140,80),(139,10,190,80),(189,10,250,80)], 400,200)
    assert t.x2 == 255 and t.iterations == 3 and t.converged


def test_no_shrink_from_paper_equations():
    t = expand_tile(Tile(0,0,100,100), [(90,20,130,60)], 300,200)
    assert t.x1 == t.y1 == 0 and t.y2 == 100


def test_clip_outside_image():
    t = expand_tile(Tile(0,0,100,100), [(-20,-10,300,400)], 250,200)
    assert t.bbox == (0,0,250,200) and t.converged


def test_empty_coarse_is_regular_grid():
    assert adaptive_tiles(3840,2160,[]) == regular_grid(3840,2160)


def test_oversize_object_is_not_silently_capped():
    t = expand_tile(Tile(0,0,1280,1280), [(0,0,3839,2159)], 3840,2160)
    assert t.bbox == (0,0,3840,2160) and t.converged


def test_iteration_limit_reports_incomplete():
    t = expand_tile(Tile(0,0,100,100), [(90,10,140,80),(139,10,190,80)], 400,200, max_iterations=1)
    assert not t.converged


def test_grid_covers_all_pixels_and_matches_sahi():
    from sahi.slicing import get_slice_bboxes
    for w,h in [(3840,2160),(1281,1281),(800,400)]:
        ours = [list(t.bbox) for t in regular_grid(w,h)]
        official = get_slice_bboxes(h,w,slice_height=1280,slice_width=1280,
                                   overlap_height_ratio=.2,overlap_width_ratio=.2,auto_slice_resolution=False)
        assert ours == official


@pytest.mark.parametrize("args", [(0,20,10,.2),(10,10,0,.2),(10,10,5,1)])
def test_invalid_grid(args):
    with pytest.raises(ValueError): regular_grid(*args)
