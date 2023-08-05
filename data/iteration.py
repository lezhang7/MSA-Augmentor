import os
import json
import numpy as np

def fetch_best_generation(result_dir):
    # fetch scores for an iteration
    # input:
    #    result_dir '/user/sunsiqi/zl/T5/AF2TEST/CASP14/output/msa_l1_u50/predict/Gtime08-19-10:48_Rpen1_Gtimes1_f_0'
    # outpout:
    #    highest_collection_path: List[file_path]
    Gsteps=os.listdir(result_dir)
    caspfile_score={}
    caspfiles=os.listdir(os.path.join(result_dir,Gsteps[0]))
    keys=[example for example in caspfiles]
    for key in keys:
        caspfile_score.update({key:[]})
    for gstep in Gsteps:
        Gstep_path=os.path.join(result_dir,gstep)
        for caspfile in caspfiles:
            caspfile_ranking_path=os.path.join(Gstep_path,caspfile,'ranking_debug.json')
            try:
                scores=json.load(open(caspfile_ranking_path,'r'))
                score=scores['plddts'][scores['order'][0]]
                caspfile_score[caspfile].append((gstep,score))
            except Exception:
                pass
    for caspfile in caspfile_score:
        total_path1=result_dir.split('predict')[0]+'total_1'
        total_path2=result_dir.split('predict')[0]+'total_2'
        score_path1=os.path.join(total_path1,caspfile.replace('generate','all'),'ranking_debug.json')
        score_path2=os.path.join(total_path2,caspfile.replace('generate','all'),'ranking_debug.json')
        score_total1=json.load(open(score_path1,'r'))['plddts'][scores['order'][0]] if os.path.exists(score_path1) else 0
        score_total2=json.load(open(score_path2,'r'))['plddts'][scores['order'][0]] if os.path.exists(score_path2) else 0    
        caspfile_score[caspfile].append(('total',max(score_total1,score_total2)))
                
    # sort and retain highest gstep for each casp file
    caspfile_step={}
    def highest_gstep(g_scores):
        gsteps=[x[0] for x in g_scores]
        scores=np.array([x[1] for x in g_scores])
        idx=np.argmax(scores)
        return gsteps[idx]
    for key in caspfile_score:
        g_scores=caspfile_score[key]
        best_step=highest_gstep(g_scores)
        caspfile_step[key]=best_step
    input_dir=result_dir.replace('output','input')
    highest_collection_path=[os.path.join(input_dir,gstep,caspfile+'.a3m') if gstep!='total' else os.path.join(input_dir.split('predict')[0],gstep,caspfile.replace('generate','all')+'.a3m') for caspfile,gstep in caspfile_step.items()]
    for path in highest_collection_path:
        assert os.path.exists(path)
    return highest_collection_path
if __name__=="__main__":
    for i in fetch_best_generation('/user/sunsiqi/zl/T5/AF2TEST/CASP14/output/msa_l1_u50/predict/Gtime08-17-08:50_Rpen1_Gtimes5_f0/'):
        assert os.path.exists(i)
        print(i)
    

