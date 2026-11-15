"""Publish a verified full-run export with explicit, scoped visual adjudications.

Does not modify the automatic run or its checkpoints. All replacements must
point to saved detector observations, and all uncertain positions stay separate.
"""
import copy,csv,gzip,hashlib,json,os,shutil
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN=Path(r'H:\data\ball-tracking-results\first_half\correction_recall_v1')
OUT=ROOT.parent/'outputs/full-correction'
ACCEPTED={'selected_observation','visually_reviewed_observation'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    complete=json.loads((RUN/'job_complete.json').read_text(encoding='utf-8'))
    assert complete['frames']==79467
    config=json.loads((RUN/'run_config.json').read_text(encoding='utf-8'))
    audit_file=ROOT/'full-run-adjudications.json'
    audit=json.loads(audit_file.read_text(encoding='utf-8')) if audit_file.exists() else {'decisions':[]}
    actions={x['source_frame']:x for x in audit['decisions']}
    assert len(actions)==len(audit['decisions'])
    for payload,expected in {(x['source_chunk_decisions'],x['source_chunk_sha256']) for x in audit['decisions']}:
        assert sha(Path(payload))==expected
    OUT.mkdir(exist_ok=True)
    refined=RUN/'qa_adjusted';refined.mkdir(exist_ok=True)
    csv_path=OUT/'corrected_ball_coordinates.csv'
    decision_path=refined/'frame_decisions.jsonl.gz'
    fields=['source_frame','part_frame','video_seconds','status','pixel_x','pixel_y','pitch_x_normalized','pitch_y_normalized','pitch_x_m','pitch_y_m','confidence','origin','candidate_index','needs_review','visual_reviewed','qa_action','quality_gate_reason','provisional_pixel_x','provisional_pixel_y','original_candidate_count','recovered_candidate_count']
    counts=Counter();orig_count=new_count=0;accepted_new=accepted_new_original_empty=0;zero_now_with_candidates=0;selected_flagged=0;applied=[];total=0;quality_gated_frames=0
    queue=[];interval=None
    with gzip.open(RUN/'frame_decisions.jsonl.gz','rt',encoding='utf-8') as src,csv_path.open('w',encoding='utf-8-sig',newline='') as f,gzip.open(decision_path,'wt',encoding='utf-8',compresslevel=5) as dst:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for line in src:
            row=json.loads(line);frame=row['source_frame'];assert frame==7171+total
            sel=row['selection'];candidates=row['candidates'];action=actions.get(frame)
            # Temporal smoothness alone was insufficient for low-confidence
            # close-up logos/hands in risk-biased QA. This is an evidence gate,
            # not a declaration that all no-H observations are false.
            if sel['status'] in ACCEPTED:
                current_candidate=candidates[sel['candidate_index']]
                insufficient_no_h=(current_candidate.get('pitch') is None and (current_candidate['confidence']<.60 or current_candidate.get('green_fraction',0.)<.25))
                if insufficient_no_h:
                    row['pre_quality_gate_selection']=copy.deepcopy(sel)
                    sel.update(status='review_candidate',provisional_pixel_x=sel['pixel_x'],provisional_pixel_y=sel['pixel_y'],pixel_x=None,pixel_y=None,needs_review=True,quality_gate_reason='no_h_and_weak_or_unresolved_closeup_identity_requires_review')
                    quality_gated_frames+=1
            # Exact visual observations can override this conservative gate;
            # an automatic threshold must not erase directly reviewed evidence.
            if action:
                row['pre_qa_selection']=copy.deepcopy(sel);row['qa_adjudication']=action
                if action['action']=='replace_with_existing_candidate_index':
                    index=action['candidate_index'];c=candidates[index]
                    assert c['point']==action['candidate_point']
                    sel.update(candidate_index=index,status='visually_reviewed_observation',pixel_x=c['point'][0],pixel_y=c['point'][1],provisional_pixel_x=None,provisional_pixel_y=None,candidate_confidence=c['confidence'],origin=c.get('origin','original_detector'),needs_review=False,visual_reviewed=True,reviewer='assistant_visual',evidence={'qa_reason':action['reason'],'automatic_path_score_not_applicable':True})
                elif action['action']=='demote_to_review':
                    sel.update(provisional_pixel_x=sel.get('pixel_x') if sel.get('pixel_x') is not None else sel.get('provisional_pixel_x'),provisional_pixel_y=sel.get('pixel_y') if sel.get('pixel_y') is not None else sel.get('provisional_pixel_y'),status='review_candidate',pixel_x=None,pixel_y=None,needs_review=True,visual_reviewed=True,reviewer='assistant_visual',qa_reason=action['reason'])
                else:raise ValueError('Unknown explicit adjudication')
                applied.append(frame)
            item=dict(source_frame=frame,part_frame=total,video_seconds=frame/25,status=sel['status'],pixel_x=sel.get('pixel_x'),pixel_y=sel.get('pixel_y'),confidence=sel.get('candidate_confidence'),origin=sel.get('origin'),candidate_index=sel['candidate_index'],needs_review=sel.get('needs_review',True),visual_reviewed=sel.get('visual_reviewed',False),qa_action=action['action'] if action else '',quality_gate_reason=sel.get('quality_gate_reason',''),provisional_pixel_x=sel.get('provisional_pixel_x'),provisional_pixel_y=sel.get('provisional_pixel_y'),original_candidate_count=row['original_candidate_count'],recovered_candidate_count=sum(c.get('origin')!='original_detector' for c in candidates))
            originals=[c for c in candidates if c.get('origin')=='original_detector']
            assert len(originals)==row['original_candidate_count']
            assert sorted(c['original_candidate_index'] for c in originals)==list(range(len(originals)))
            if sel['status'] in ACCEPTED:
                c=candidates[sel['candidate_index']]
                assert [item['pixel_x'],item['pixel_y']]==c['point']
                assert item['confidence']==c['confidence']
                if 'pitch' in c:
                    x,y=c['pitch'];item.update(pitch_x_normalized=x,pitch_y_normalized=y,pitch_x_m=x*105.3,pitch_y_m=y*68)
                if c.get('origin')!='original_detector':
                    accepted_new+=1
                    if not originals:accepted_new_original_empty+=1
                selected_flagged+=bool(sel.get('needs_review'))
            else:assert item['pixel_x'] is None and item['pixel_y'] is None
            if not originals and candidates:zero_now_with_candidates+=1
            writer.writerow(item);dst.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
            counts[sel['status']]+=1;orig_count+=len(originals);new_count+=item['recovered_candidate_count'];total+=1
            reasons=[];e=sel.get('evidence') or {}
            if sel['status'] not in ACCEPTED:reasons.append(sel['status'])
            if sel.get('needs_review'):reasons.append('review_flag')
            for key in ('atd_conflict','competing_candidate','camera_motion_uncertain','near_camera_cut'):
                if e.get(key):reasons.append(key)
            if action and action['action']=='demote_to_review':reasons.append('visual_identity_or_localization_concern')
            uncertain=sel['status'] not in ACCEPTED or sel.get('needs_review',True)
            if uncertain:
                if interval is None:interval=dict(start_frame=frame,end_frame_exclusive=frame+1,reason_counts=Counter())
                interval['end_frame_exclusive']=frame+1;interval['reason_counts'].update(reasons)
            elif interval is not None:queue.append(interval);interval=None
    if interval is not None:queue.append(interval)
    assert total==79467 and orig_count==244334
    assert sorted(applied)==sorted(actions)
    with (OUT/'review_intervals.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['start_frame','end_frame_exclusive','start_video_seconds','duration_seconds','reasons'])
        for q in sorted(queue,key=lambda x:x['end_frame_exclusive']-x['start_frame'],reverse=True):writer.writerow([q['start_frame'],q['end_frame_exclusive'],q['start_frame']/25,(q['end_frame_exclusive']-q['start_frame'])/25,json.dumps(q['reason_counts'])])
    with csv_path.open('rb') as f,gzip.open(OUT/'corrected_ball_coordinates.csv.gz','wb',compresslevel=6) as g:shutil.copyfileobj(f,g)
    shutil.copy2(csv_path,refined/csv_path.name)
    if audit_file.exists():shutil.copy2(audit_file,OUT/'visual_adjudications.json')
    shutil.copy2(RUN/'run_config.json',OUT/'run_config.json')
    summary=dict(frames=total,source_frame_range=[7171,86638],source_mp4_time=['00:04:46.84','00:57:45.52'],automatic_phase=complete,final_status_counts=dict(counts),accepted_frames=sum(counts[k] for k in ACCEPTED),selected_frames_still_flagged_for_review=selected_flagged,original_candidates_retained=orig_count,new_candidate_observations=new_count,selected_recovered_frames=accepted_new,selected_frames_previously_without_any_original_candidate=accepted_new_original_empty,frames_with_candidates_after_previously_zero=zero_now_with_candidates,low_confidence_no_h_evidence_gate_applied=quality_gated_frames,quality_gate_threshold=.60,quality_gate_is_not_false_positive_label=True,applied_visual_adjudications=len(applied),applied_exact_frames=applied,review_intervals=len(queue),accuracy_validated=False,no_interpolation_or_copied_atd_coordinates=True,csv_sha256=sha(csv_path),qa_decisions_sha256=sha(decision_path),qa_decisions_path=str(decision_path),code_sha256=sha(Path(__file__)))
    summary['no_h_evidence_gate_applied']=summary.pop('low_confidence_no_h_evidence_gate_applied')
    summary['quality_gate']=dict(confidence_threshold=.60,object_center_green_threshold=.25,rule='No pitch projection and (confidence below .60 or center-patch green fraction below .25) routes automatic selection to review; exact visual approvals apply afterward.',not_a_false_positive_classifier=True)
    summary.pop('quality_gate_threshold',None)
    (OUT/'completion_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (refined/'job_complete.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
