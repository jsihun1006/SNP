"""Aggregate, assemble and audit the separate full-match diagnostic artifacts."""

import argparse
from collections import Counter,defaultdict
import csv
import datetime
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

import tryolabs_full_match_v2 as full

OUT=full.OUT
REPORT=full.ROOT/"docs/TRYOLABS_BROADCAST_FULL_MATCH_V2.md"


def result_rows(out):
    db=full.connection(out)
    try:
        for i,b in db.execute("SELECT frame0,payload FROM results ORDER BY frame0"):
            yield full.unpack(b)
    finally:db.close()


def coverage(out):
    db=full.connection(out);manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    expected=set(range(manifest["start"],manifest["end"]))
    detected={i for i, in db.execute("SELECT frame0 FROM detections")}
    processed={i for i, in db.execute("SELECT frame0 FROM results")}
    bad=[];origins=Counter()
    for i,b in db.execute("SELECT frame0,payload FROM detections"):
        raw=full.unpack(b);origins[raw.get("origin","unknown")]+=1
        if raw["status"]!="ok":bad.append(i)
    result=dict(expected_frames=len(expected),detected_frames=len(expected&detected),processed_frames=len(expected&processed),
                detector_missing_ranges=full.ranges(expected-detected),unprocessed_ranges=full.ranges(expected-processed),
                detector_failed_ranges=full.ranges(bad),detector_origins=dict(origins),complete=expected<=processed and not bad)
    db.close();return result


def hist_add(hist,value):
    hist[min(1000,max(0,int(round(value*1000))))]+=1


def hist_summary(hist):
    total=int(hist.sum())
    if not total:return dict(n=0)
    cumulative=np.cumsum(hist)
    return dict(n=total,mean=float(np.dot(hist,np.arange(1001)/1000)/total),
                p10=float(np.searchsorted(cumulative,.1*total)/1000),p50=float(np.searchsorted(cumulative,.5*total)/1000),
                p90=float(np.searchsorted(cumulative,.9*total)/1000))


def consecutive_runs(rows,predicate):
    result=[];start=None;last=None;scene=None
    for r in rows:
        valid=predicate(r)
        if start is not None and (not valid or r["frame0"]!=last+1 or r["scene_id"]!=scene):
            result.append(dict(start_frame0=start,end_frame0=last,frames=last-start+1,seconds=(last-start+1)/25,scene_id=scene));start=None
        if valid:
            if start is None:start=r["frame0"];scene=r["scene_id"]
            last=r["frame0"]
    if start is not None:result.append(dict(start_frame0=start,end_frame0=last,frames=last-start+1,seconds=(last-start+1)/25,scene_id=scene))
    return result


def summarize_runs(rr):
    lengths=[r["frames"] for r in rr]
    if not lengths:return dict(runs=0)
    return dict(runs=len(rr),frames=sum(lengths),max_frames=max(lengths),p50_frames=float(np.median(lengths)),
                p90_frames=float(np.quantile(lengths,.9)),longest=sorted(rr,key=lambda r:-r["frames"])[:20])


def aggregate(out):
    cov=coverage(out);rows=list(result_rows(out))
    total=Counter();bins=defaultdict(Counter);reason=Counter();views=defaultdict(Counter);sizes=defaultdict(Counter)
    score_hists=defaultdict(lambda:np.zeros(1001,dtype=np.int64));classes=Counter();old_classes=Counter()
    previous={};ref_tracks={};ref_episodes=[];changes=Counter();suspects=[]
    margin_hist=np.zeros(1001,dtype=np.int64);ball_hist=np.zeros(1001,dtype=np.int64)
    by_frame={r["frame0"]:r for r in rows};failures=[];cut_validation=[];last=None
    for r in rows:
        bucket=r["frame0"]//15000;b=bins[bucket];b["frames"]+=1;total["frames"]+=1
        for k,v in r["v2_classes"].items():b[k]+=v
        b["objects"]+=r["objects"];reason.update(r["unknown_reasons"])
        ball_key="ball0" if r["ball_count"]==0 else ("ball1" if r["ball_count"]==1 else "ball2plus")
        b[ball_key]+=1;total[ball_key]+=1
        b[r["evidence_status"]]+=1;total[r["evidence_status"]]+=1
        total["referee_exclusion_frames"]+=r["referees"]>0;b["referee_frames"]+=r["referees"]>0
        total["referee_nearest_excluded_frames"]+=r["referee_nearest_excluded"]
        total["missing_ball_track_frames"]+=r["main_ball_age"] is None
        total["stale_main_ball_frames"]+=r["main_ball_age"] is not None and r["main_ball_age"]>0
        total["camera_cut_suspect_frames"]+=r["diagnostics"]["cut_suspect"]
        total["repeated_sequence_suspect_samples"]+=r["diagnostics"]["repeated_sequence_suspect"]
        total["off_field_suspect_frames"]+=r["diagnostics"]["off_field_suspect"]
        if r["reset_reason"]:total[f'reset_{r["reset_reason"]}']+=1
        for conf in r["ball_confidences"]:hist_add(ball_hist,conf)
        for field in ["motion_error","v1_classification_error","v1_match_error"]:
            if r.get(field):failures.append(dict(frame0=r["frame0"],kind=field,error=r[field]));total[field]+=1
        if last is not None and last["scene_id"]==r["scene_id"]:
            total["v1_candidate_changes"]+=r["v1_candidate_id"]!=last["v1_candidate_id"]
            total["v2_candidate_changes"]+=r["candidate_id"]!=last["candidate_id"]
        if r["diagnostics"]["cut_suspect"] and last is not None and r["scene_id"]==last["scene_id"]:
            cut_validation.append(r["frame0"])
        total["v1_v2_candidate_different_frames"]+=r["v1_candidate_id"]!=r["candidate_id"]
        last=r
    for folder in sorted((out/"chunks").glob("[0-9]*")):
        with gzip.open(folder/"objects.jsonl.gz","rt",encoding="utf-8") as f:
            for line in f:
                frame=json.loads(line);i=frame["frame0"]
                if i not in by_frame:continue
                view=by_frame[i]["diagnostics"]["view"]
                for o in frame["objects"]:
                    label=o["label"];classes[label]+=1;old_classes[o["legacy_class"]]+=1;views[view][label]+=1
                    size="SMALL_LT60PX" if o["bbox_height"]<60 else ("MEDIUM_60_160PX" if o["bbox_height"]<160 else "LARGE_GE160PX")
                    sizes[size][label]+=1
                    hist_add(margin_hist,o.get("score_margin",0.))
                    for k,v in o["scores"].items():hist_add(score_hists[k],v)
                    for k,v in o["components"].items():hist_add(score_hists["component_"+k],v)
                    key=(frame["scene_id"],o["track_id"]);prev=previous.get(key)
                    if prev and i==prev[0]+1:
                        changes["v1_class_changes"]+=o["legacy_class"]!=prev[1]
                        if label!=prev[2]:
                            changes["v2_class_changes"]+=1
                            changes["v2_unknown_boundary_changes"]+="UNKNOWN" in [label,prev[2]]
                        changes["referee_to_unknown"]+=prev[2]=="REFEREE" and label=="UNKNOWN"
                    previous[key]=(i,o["legacy_class"],label)
                    if o["eligible"] and (o["role"]!="PLAYER" or o["team"] not in ("TEAM_A","TEAM_B") or o["age"]):
                        raise AssertionError(f'Invalid eligible object at {i}')
                    if o["role"]=="REFEREE":
                        track=ref_tracks.setdefault(key,dict(scene_id=key[0],track_id=key[1],start_frame0=i,end_frame0=i,observed_frames=0,edge_suspect_frames=0))
                        track["end_frame0"]=i;track["observed_frames"]+=1
                        (x1,y1),(x2,y2)=o["points"]
                        edge=(x1+x2)/2<.1*1920 or (x1+x2)/2>.9*1920 or y2>.92*1080
                        if edge:
                            track["edge_suspect_frames"]+=1
                            suspects.append(dict(frame0=i,scene_id=key[0],track_id=key[1],reason="REFEREE_APPEARANCE_AT_IMAGE_EDGE_NOT_STAFF_TRUTH"))
    for t in ref_tracks.values():
        t["observed_seconds"]=t["observed_frames"]/25;t["span_seconds"]=(t["end_frame0"]-t["start_frame0"]+1)/25
    ball_runs=consecutive_runs(rows,lambda r:r["ball_count"]>=1)
    unique_runs=consecutive_runs(rows,lambda r:r["ball_count"]==1)
    proximity_runs=consecutive_runs(rows,lambda r:r["candidate_team"] is not None)
    evidence_runs=consecutive_runs(rows,lambda r:r["evidence_status"]=="PROXIMITY_20_FRAME_EVIDENCE")
    for name,value in [("ball_observed_runs",ball_runs),("ball_unique_runs",unique_runs),("proximity_runs",proximity_runs),("evidence20_runs",evidence_runs),("referee_tracks",list(ref_tracks.values())),("referee_edge_suspects",suspects),("runtime_failures",failures)]:
        full.atomic_json(out/f"{name}.json",value)
    windows=[(15000,15250),(45000,45250),(90000,90250)]
    sample_frames={i for a,b in windows for i in range(a,b)};sample_classes=Counter();sample_status=Counter()
    for r in rows:
        if r["frame0"] in sample_frames:sample_classes.update(r["v2_classes"]);sample_status[r["evidence_status"]]+=1
    old_path=full.legacy.ASSETS/"classification_v2/summary.json"
    old=json.loads(old_path.read_text(encoding="utf-8")) if old_path.is_file() else None
    original_750=(dict(available=True,classes=old["total_v2_classes"],unknown_fraction=old["unknown_fraction"])
                  if old is not None else dict(available=False,note="Prior private Mac results are optional and were not supplied"))
    summary=dict(coverage=cov,totals=dict(total),v2_classes=dict(classes),v1_classes=dict(old_classes),
                 unknown_fraction=classes["UNKNOWN"]/max(1,sum(classes.values())),unknown_reasons=dict(reason),
                 time_bins=[dict(bin=k,start_seconds=k*600,end_seconds=min((k+1)*600,full.TOTAL/25),**v) for k,v in sorted(bins.items())],
                 views={k:dict(v) for k,v in views.items()},size_bins={k:dict(v) for k,v in sizes.items()},
                 scores={k:hist_summary(v) for k,v in score_hists.items()},score_margin=hist_summary(margin_hist),ball_confidence=hist_summary(ball_hist),
                 class_changes=dict(changes),referee_tracks=len(ref_tracks),scene_ids=len({r["scene_id"] for r in rows}),
                 shot_reset_validation_failures=cut_validation,ball_observed_runs=summarize_runs(ball_runs),ball_unique_runs=summarize_runs(unique_runs),
                 proximity_runs=summarize_runs(proximity_runs),evidence20_runs=summarize_runs(evidence_runs),
                 original_750=original_750,
                 same_750_with_full_context=dict(processed_frames=sum(r["frame0"] in sample_frames for r in rows),classes=dict(sample_classes),evidence_statuses=dict(sample_status),
                                                 note="same raw detector cache; tracking context and scene-reset policy differ from 3 isolated samples"),
                 live_play_verified=False,actual_replay_verified=False,metrics_are_accuracy=False)
    full.atomic_json(out/"summary.json",summary)
    with (out/"time_bins.csv").open("w",encoding="utf-8",newline="") as f:
        fields=sorted({k for r in summary["time_bins"] for k in r})
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary["time_bins"])
    return summary


def concat_entry(path):
    # FFmpeg concat uses forward slashes even for native Windows drive paths.
    value=Path(path).resolve().as_posix().replace("'", "'\\''")
    return f"file '{value}'\n"


def assemble(out):
    cov=coverage(out)
    if not cov["complete"]:raise RuntimeError(f'Full processing incomplete: {cov["unprocessed_ranges"]}')
    destination=out/"comparison_full.mp4"
    if not destination.exists():
        paths=sorted((out/"chunks").glob("[0-9]*/comparison.mp4"))
        listing=out/"concat.txt"
        listing.write_text("".join(concat_entry(p) for p in paths),encoding="utf-8")
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-nostdin","-v","error","-n","-f","concat","-safe","0","-i",str(listing),"-c","copy","-movflags","+faststart",str(destination)],check=True)
    decoded,fps,size=full.decode_count(destination)
    expected=cov["expected_frames"]
    if decoded!=expected or fps!=25. or size!=(full.WIDTH,full.HEIGHT):raise ValueError("Full video decoder count/fps/size mismatch")
    full.atomic_json(out/"video_verification.json",dict(decoded_frames=decoded,fps=fps,size=size,sha256=full.legacy.digest(destination),passed=True))


def select_clips(rows):
    choices=[]
    criteria=[("evidence20",lambda r:r["evidence_status"]=="PROXIMITY_20_FRAME_EVIDENCE"),
              ("unknown_wide",lambda r:r["diagnostics"]["view"]=="WIDE_HYPOTHESIS" and r["objects"]>=8 and r["v2_classes"].get("UNKNOWN",0)/r["objects"]>.8),
              ("multiple_balls",lambda r:r["ball_count"]>=3),
              ("stale_ball",lambda r:r["main_ball_age"] is not None and r["main_ball_age"]>10),
              ("referee_closeup",lambda r:r["referees"]>0 and r["diagnostics"]["view"]=="CLOSEUP_HYPOTHESIS"),
              ("repeated_visual_sequence",lambda r:r["diagnostics"]["repeated_sequence_suspect"]),
              ("off_field_suspect",lambda r:r["diagnostics"]["off_field_suspect"]),
              ("ball_missing_wide",lambda r:r["ball_count"]==0 and r["diagnostics"]["view"]=="WIDE_HYPOTHESIS")]
    for name,predicate in criteria:
        runs=consecutive_runs(rows,predicate)
        if not runs:continue
        best=max(runs,key=lambda r:r["frames"])
        center=(best["start_frame0"]+best["end_frame0"])//2
        start=max(0,min(full.TOTAL-250,center-125))
        if any(abs(start-c["start_frame0"])<250 for c in choices):continue
        choices.append(dict(category=name,start_frame0=start,end_frame0=start+249,frames=250,trigger_run=best,
                            interpretation="Automatically selected diagnostic hypothesis, no human event labels"))
    return choices


def extract_clips(out):
    if not (out/"video_verification.json").exists():raise ValueError("Assemble and verify the full video first")
    choices=select_clips(list(result_rows(out)));folder=out/"examples";folder.mkdir(exist_ok=True)
    import imageio_ffmpeg
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    for c in choices:
        destination=folder/f'{c["category"]}_{c["start_frame0"]}.mp4';c["path"]=str(destination.relative_to(out))
        if not destination.exists():
            subprocess.run([exe,"-nostdin","-v","error","-n","-ss",str(c["start_frame0"]/25),"-i",str(out/"comparison_full.mp4"),"-frames:v","250","-an","-c:v","libx264","-preset","veryfast","-crf","21","-pix_fmt","yuv420p",str(destination)],check=True)
        count,_,_=full.decode_count(destination)
        if count!=250:raise ValueError(f"Example video frame mismatch: {destination}")
        cap=cv2.VideoCapture(str(destination));cap.set(cv2.CAP_PROP_POS_FRAMES,125);ok,im=cap.read();cap.release()
        if ok:cv2.imwrite(str(destination.with_suffix('.jpg')),im)
    full.atomic_json(out/"examples.json",choices)


def verify_protected(out):
    snapshot=out/"preservation_before.json"
    if not snapshot.is_file():snapshot=OUT/"preservation_before.json"  # Preserve legacy Mac audit support.
    before=json.loads(snapshot.read_text(encoding="utf-8"))
    changed=[path for path,entry in before["files"].items() if not (full.ROOT/path).is_file() or full.legacy.digest(full.ROOT/path)!=entry["sha256"]]
    st=full.VIDEO.stat();video_same=(st.st_size,st.st_mtime_ns)==(before["video"]["size"],before["video"]["mtime_ns"])
    if changed or not video_same:raise ValueError(f"Protected input changed: {changed}")
    result=dict(protected_files=len(before["files"]),changed_files=changed,source_video_size_mtime_unchanged=video_same)
    full.atomic_json(out/"preservation_after.json",result);return result


def plots(out,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins=summary["time_bins"];x=[r["start_seconds"]/60 for r in bins]
    fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
    ax[0].plot(x,[r.get("UNKNOWN",0)/max(1,r.get("objects",0)) for r in bins],marker="o");ax[0].set_ylabel("UNKNOWN / objects")
    ax[1].plot(x,[r.get("ball1",0)/max(1,r["frames"]) for r in bins],marker="o");ax[1].set_ylabel("Exactly one ball / frames")
    ax[2].plot(x,[r.get("PROXIMITY_20_FRAME_EVIDENCE",0)/max(1,r["frames"]) for r in bins],marker="o");ax[2].set_ylabel("20-frame evidence / frames");ax[2].set_xlabel("Broadcast file time (minutes)")
    for a in ax:a.grid(alpha=.3);a.set_ylim(0,1)
    fig.suptitle("Full broadcast diagnostic rates (not accuracy; live play unverified)")
    fig.tight_layout();fig.savefig(out/"time_diagnostics.png",dpi=150);plt.close(fig)


def write_report(out,s, report=REPORT):
    cov=s["coverage"];t=s["totals"];n=max(1,t.get("frames",0));objects=sum(s["v2_classes"].values())
    complete=cov["complete"] and (out/"video_verification.json").exists()
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    lines=["# Tryolabs Broadcast 전체 영상 V2 진단", "",f"기록 시각(KST): {now.isoformat(timespec='seconds')}","",
           f"**전체 작업 완료 여부: {'완료' if complete else '미완료 — 실행 중 또는 미처리 범위 있음'}**.",
           f"목표 {cov['expected_frames']:,}프레임 / 검출 캐시 {cov['detected_frames']:,} / 추적·분류·조각 영상 검증 {cov['processed_frames']:,}.",
           f"미처리 범위(양 끝 포함,0-based): `{cov['unprocessed_ranges']}`.",f"검출 실패 범위: `{cov['detector_failed_ranges']}`.","",
           "기존 V1/V2 코드·HSV 설정·정책·가중치와 결과는 보존했다. 같은640/BGR 검출 결과를 V1/V2가 공유한다.",
           f"공유 YOLOv5 FP32 실행 장치: {manifest['device']}, 요청 배치: {manifest['batch']}, 모델 상주 정책: {manifest.get('model_residency','legacy')}. NMS는 CPU에서 프레임별 실행한다.",
           f"기존750프레임 캐시 재사용 설정: {manifest.get('reuse_750_cache','legacy')}. 실제 캐시 origin과 runtime_events.jsonl을 함께 확인한다.",
           "공 모델·유니폼 설정·45px/20프레임 기준을 바꾸거나 새로 학습하지 않았다. 장면 불연속의 외부 안전 초기화만 추가했다.",
           "정답 라벨이 없으므로 아래 수치는 Accuracy/Precision/Recall/F1이 아니다. TEAM은 appearance 이름이며 실제 H/A 소속 확정이 아니다.","",
           "## 전체 및 시간대별 통계","",f"V2 객체 관측 수: {objects:,}; UNKNOWN {s['unknown_fraction']:.2%}.",
           f"V2 클래스: `{s['v2_classes']}`. V1 클래스: `{s['v1_classes']}`.",f"UNKNOWN 주요 원인: `{s['unknown_reasons']}`.",
           f"동일 scene/track의 인접 관측 변경: `{s['class_changes']}`. UNKNOWN 진입·복귀를 포함하며 ID는 영구 인물 ID가 아니다.","",
           "| 파일 시간(분) | 프레임 | 객체 | UNKNOWN | 공0개 | 공1개 | 공2개 이상 | 20프레임 근거 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in s["time_bins"]:
        lines.append(f"| {r['start_seconds']/60:.0f}–{r['end_seconds']/60:.2f} | {r['frames']} | {r.get('objects',0)} | {r.get('UNKNOWN',0)/max(1,r.get('objects',0)):.1%} | {r.get('ball0',0)} | {r.get('ball1',0)} | {r.get('ball2plus',0)} | {r.get('PROXIMITY_20_FRAME_EVIDENCE',0)} |")
    lines += ["",f"화면 종류별(휴리스틱): `{s['views']}`.",f"선수 박스 높이별 분포: `{s['size_bins']}`.",
              "SMALL<60px, MEDIUM60–160px, LARGE≥160px은 원본1080p 박스 높이 기준이다.",
              f"HSV 지지율·점수 분포(0.001 단위 histogram 분위수): `{s['scores']}`.",f"1·2위 점수 차이: `{s['score_margin']}`. 지지율과 점수는 정답 확률이 아니다.","",
              "## 심판과 점유 후보 제외","",f"REFEREE 객체 관측 {s['v2_classes'].get('REFEREE',0):,}건, 해당 scene/track 조합 {s['referee_tracks']:,}개.",
              f"REFEREE가 후보에서 제외된 프레임 {t.get('referee_exclusion_frames',0):,}; 최근접 REFEREE 제외 프레임 {t.get('referee_nearest_excluded_frames',0):,}.",
              "지속시간은 referee_tracks.json의 관측 초와 최초–최종 span을 구분한다. 중간 UNKNOWN·가림은 실제 지속 관측으로 세지 않는다.",
              "화면 가장자리의 REFEREE appearance는 referee_edge_suspects.json에 남긴다. 이는 스태프 오인식 의심 후보이며 실제 심판/스태프 정답이 아니다.","",
              "## 공 검출과 점유 근거","",f"공0개/1개/2개 이상: {t.get('ball0',0):,} ({t.get('ball0',0)/n:.2%}) / {t.get('ball1',0):,} ({t.get('ball1',0)/n:.2%}) / {t.get('ball2plus',0):,} ({t.get('ball2plus',0)/n:.2%}).",
              f"공 후보 confidence 분포: `{s['ball_confidence']}`. 후보가 하나여도 실제 공임을 보증하지 않는다.",
              f"main ball track 누락 {t.get('missing_ball_track_frames',0):,}, stale {t.get('stale_main_ball_frames',0):,}프레임.",
              f"연속 공 관측: `{s['ball_observed_runs']}`.",f"연속 단일 공 후보: `{s['ball_unique_runs']}`.",
              f"PROXIMITY_PENDING {t.get('PROXIMITY_PENDING',0):,}; PROXIMITY_20_FRAME_EVIDENCE {t.get('PROXIMITY_20_FRAME_EVIDENCE',0):,}; INSUFFICIENT_EVIDENCE {t.get('INSUFFICIENT_EVIDENCE',0):,} ({t.get('INSUFFICIENT_EVIDENCE',0)/n:.2%}).",
              f"후보 변경 V1 {t.get('v1_candidate_changes',0):,}, V2 {t.get('v2_candidate_changes',0):,}; 후보가 다른 frame {t.get('v1_v2_candidate_different_frames',0):,}. None 전환을 포함하며 scene을 이어 붙이지 않는다.",
              f"후보 근거 연속 길이: `{s['proximity_runs']}`.",f"20연속 조건 이후 근거 길이: `{s['evidence20_runs']}`.",
              "evidence20_runs.json에 알고리즘 조건이 성립한 정확한 source 범위를 저장한다. **실제 controlled possession이 확인됐다는 뜻은 아니다.** 경기 진행 여부는 전 프레임 UNVERIFIED다.","",
              "## 컷·리플레이·경기 외 화면과 상태 초기화","",
              f"컷 의심 {t.get('camera_cut_suspect_frames',0):,}, 반복 시퀀스 의심 표본 {t.get('repeated_sequence_suspect_samples',0):,}, 경기장 외 의심 {t.get('off_field_suspect_frames',0):,}, 관측 scene 수 {s['scene_ids']:,}.",
              f"컷 의심인데 scene 상태가 초기화되지 않은 frame: `{s['shot_reset_validation_failures']}`.",
              "컷은 축소 영상의 HSV histogram/밝기 변화와 perceptual hash로 감지한다. 반복 화면은25프레임 간격의 시각 hash가 이전 순서대로3회 반복되는지 검사한다.",
              "잔디색 비율·사람 박스 크기는 와이드/클로즈업/화면 외 의심 진단용이다. 같은 색상의 카메라 컷 누락, 정적인 화면의 반복 오인 가능성이 있다.",
              "실제 리플레이나 경기 중/정지 상태를 확정하지 않으며, 확인되지 않은 구간을 임의로 경기 중으로 설정하지 않는다.",
              "감지한 컷·frame gap·운동 추정 실패에서는 두 tracker와 V1/V2 시간적 분류·근접 이력을 함께 초기화한다. 청크·재시작 경계 자체에서는 초기화하지 않고 checkpoint를 복원한다.","",
              "## 750프레임 실험과의 비교","",f"기존 고립3구간 결과: `{s['original_750']}`.",f"전체 문맥에서 같은750프레임: `{s['same_750_with_full_context']}`.",
              "동일750프레임의 검출 출처는 coverage.detector_origins와 캐시 origin을 확인한다. 이전 문맥·컷 초기화·추론 장치가 다를 수 있어 class 수의 완전 일치를 요구하지 않는다. HSV 설정을 다시 맞추지 않았다.","",
              "## 오류·보존·출력","",f"런타임 진단: motion_error {t.get('motion_error',0):,}, V1 분류 오류 {t.get('v1_classification_error',0):,}, V1 Match 오류 {t.get('v1_match_error',0):,}프레임.",
              "runtime_failures.json에 frame과 오류를 기록한다. V1의 OTHER 팀으로 pass state가 실패하면 해당 실패를 표시하고 pass state만 격리하며 이미 계산한 점유값은 숨기지 않는다.",
              "검출은 run.sqlite의 detections, 프레임 요약은 results에 저장한다. 각 chunks/*/ 아래 tracking.jsonl.gz, objects.jsonl.gz, frames.jsonl.gz, comparison.mp4, state.pkl을 구분해 저장한다.",
              "청크 전체 인코딩/디코딩을 확인한 뒤 atomic rename과 DB transaction으로 완료 처리한다. 체크포인트/영상 SHA256을 기록하고 미완성 청크는 interrupted에 보존한다.",
              "재개 동작은 tests/test_tryolabs_windows.py의 합성 CPU 영상 회귀 테스트로 검사한다. 이 보고서는 Windows GPU 재개·처리량 시험을 별도로 실행했다고 주장하지 않는다.",
              f"파일 보존 검사는 이 실행 폴더의 preservation_after.json을 확인한다: {out}.","",
              "## 일반화 한계와 V3 우선순위","",
              "1. 공 후보의 다중성·누락·오검출을 먼저 분리 진단하고 외부 경기 라벨로 검증한다. 이 단계에서 공 모델은 변경하지 않았다.",
              "2. 심판과 노란 조끼 인력의 구분에는 색상 이외의 인물/포즈/경기장 문맥이 필요하다. REFEREE는 appearance 가설이다.",
              "3. UNKNOWN 원인·크기·조명·화면 종류별 지지율을 바탕으로 외부 경기의 독립 평가를 마련한다. 현재 경기의 수작업 라벨로 임계값을 튜닝하지 않는다.",
              "4. 컷·리플레이·경기 진행 상태를 검증 가능한 별도 모듈로 분리하고, 골키퍼의 역할과 팀 소속도 별도로 검증한다.",
              "고정 설정이 전체 영상에서 작동한다는 것은 실행상의 일반화일 뿐, 팀·역할 분류의 의미적 정확도나 다른 경기 성능을 보증하지 않는다."]
    if (out/"video_verification.json").exists():lines += ["",f"전체 비교 영상: `{out/'comparison_full.mp4'}`."]
    if (out/"examples.json").exists():
        lines += ["","자동 대표 구간(정상·실패는 알고리즘 조건 관점의 진단):"]
        for c in json.loads((out/"examples.json").read_text()):
            lines.append(f"- {c['category']}: `{out/c['path']}`, source frame {c['start_frame0']}..{c['end_frame0']}.")
    report=Path(report)
    if report.exists():raise FileExistsError(f"Report exists; pass a new --report path: {report}")
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=OUT/"main")
    p.add_argument("--assemble",action="store_true");p.add_argument("--clips",action="store_true");p.add_argument("--verify",action="store_true")
    p.add_argument("--report",type=Path,help="Default: a new report inside this run; existing reports are preserved")
    args=p.parse_args();out=args.output.resolve()
    if not out.is_relative_to(OUT.resolve()):p.error("Expected full_match_v2 output")
    report=args.report or out/"report.md"
    if report.exists():p.error("Report already exists; specify a new --report path")
    if args.assemble:assemble(out)
    if args.clips:extract_clips(out)
    s=aggregate(out)
    if args.verify:verify_protected(out)
    plots(out,s);write_report(out,s,report)
    print(json.dumps(s["coverage"],indent=2))


if __name__=="__main__":main()
