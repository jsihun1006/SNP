METAINFO = dict(
    dataset_name="synloc_two_points",
    paper_info=dict(title="SSS-1 independent reimplementation", author="", container="", year="2026"),
    keypoint_info={
        0: dict(name="pelvis", id=0, color=[255, 120, 0], type="lower", swap=""),
        1: dict(name="ground", id=1, color=[0, 200, 255], type="lower", swap=""),
    },
    skeleton_info={0: dict(link=("pelvis", "ground"), id=0, color=[0, 255, 0])},
    joint_weights=[1., 1.], sigmas=[.089, .089],
)
