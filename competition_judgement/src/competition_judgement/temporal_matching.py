"""Only reciprocal-unique adjacent geometric matches; no tracking identities."""
import numpy as np
import pandas as pd


def circle_iou(distance,left_radius,right_radius):
    d=np.asarray(distance,dtype=float);r=np.broadcast_to(left_radius[:,None],d.shape);s=np.broadcast_to(right_radius[None,:],d.shape)
    safe=np.maximum(d,1e-15)
    a=np.arccos(np.clip((d*d+r*r-s*s)/(2*safe*r),-1,1))
    b=np.arccos(np.clip((d*d+s*s-r*r)/(2*safe*s),-1,1))
    radical=np.maximum(0,(-d+r+s)*(d+r-s)*(d-r+s)*(d+r+s))
    area=r*r*a+s*s*b-.5*np.sqrt(radical)
    area=np.where(d>=r+s,0,np.where(d<=np.abs(r-s),np.pi*np.minimum(r,s)**2,area))
    return area/(np.pi*(r*r+s*s)-area)


def reciprocal_unique(left_xy,right_xy,left_size,right_size,left_diameter,right_diameter,config):
    d=np.linalg.norm(left_xy[:,None]-right_xy[None,:],axis=-1)
    ratio=np.minimum(left_size[:,None],right_size[None,:])/np.maximum(left_size[:,None],right_size[None,:])
    iou=circle_iou(d,np.maximum(config['matching_min_disk_radius_m'],left_diameter/2),
                  np.maximum(config['matching_min_disk_radius_m'],right_diameter/2))
    eligible=(d<=config['matching_max_center_step_m'])&(ratio>=config['matching_min_size_ratio'])&(iou>=config['matching_min_disk_iou'])
    ldegree=eligible.sum(1);rdegree=eligible.sum(0)
    pairs=[]
    for i,j in zip(*np.nonzero(eligible)):
        if ldegree[i]==1 and rdegree[j]==1:pairs.append((int(i),int(j),float(d[i,j])))
    return eligible,pairs


def adjacent_matches(source,clusters,config):
    n=len(clusters)
    columns={}
    for direction in ('prev','next'):
        columns[f'match_{direction}_status']=np.full(n,'boundary_or_outside',object)
        columns[f'match_{direction}_frame']=np.full(n,np.nan)
        columns[f'match_{direction}_cluster_id']=np.full(n,np.nan)
        columns[f'match_{direction}_center_step_m']=np.full(n,np.nan)
        columns[f'match_{direction}_candidate_count']=np.full(n,np.nan)
    xy=clusters[['cluster_centroid_x','cluster_centroid_y']].to_numpy()
    sizes=clusters.n_players.to_numpy();diameter=clusters.cluster_max_pairwise_distance_m.to_numpy();ids=clusters.cluster_id.to_numpy()
    for row in range(len(source.ids)-1):
        if source.periods[row]==0 or source.last[row]<=row:continue
        reliable=(source.strict[row] and source.strict[row+1] and source.nh[row]==source.nh[row+1] and source.na[row]==source.na[row+1])
        for radius in (0,1):
            left=source.component_rows[row,radius];left=left[left>=0]
            right=source.component_rows[row+1,radius];right=right[right>=0]
            if not reliable:
                columns['match_next_status'][left]='observation_ambiguous';columns['match_prev_status'][right]='observation_ambiguous'
                continue
            mask,pairs=reciprocal_unique(xy[left],xy[right],sizes[left],sizes[right],diameter[left],diameter[right],config)
            for selected,degree,direction in ((left,mask.sum(1),'next'),(right,mask.sum(0),'prev')):
                columns[f'match_{direction}_candidate_count'][selected]=degree
                columns[f'match_{direction}_status'][selected]=np.where(degree>0,'geometric_ambiguous','no_clear_spatial_match')
            for i,j,step in pairs:
                l,r=left[i],right[j]
                for current,target,direction,frame in ((l,r,'next',source.ids[row+1]),(r,l,'prev',source.ids[row])):
                    columns[f'match_{direction}_status'][current]='reciprocal_unique_geometry'
                    columns[f'match_{direction}_frame'][current]=frame
                    columns[f'match_{direction}_cluster_id'][current]=ids[target]
                    columns[f'match_{direction}_center_step_m'][current]=step
    return pd.DataFrame(columns)
