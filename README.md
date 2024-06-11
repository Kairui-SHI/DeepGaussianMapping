This is a preview version for editing ...   

***The code is still in the debugging phase, and there are many details such as absolute paths and mode switches that need to be adjusted!***  


## Installation
```bash
conda create -n DeepGS python=3.10
conda activate DeepGS
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Prepare for Datasets  
Download the data as below, and the data will be saved into the ./data/Replica folder.  
```bash
bash bash_scripts/download_replica.sh  
```
If you download dataset by yourself, please collect data in SplaTAM format, including `./congfigs` and `./datasets/`.  

For details on how to obtain prepared dataset and the format of the dataset, see SplaTAM:    
https://github.com/spla-tam/SplaTAM  

For details on L-net or the scripts: /scripts/deepmapping.py , see DeepMapping2:     
https://github.com/ai4ce/DeepMapping2  

## Usage   
**run SplaTAM for tracking initial poses**    
recommand first try SplaTAM to test whether the dataset is good enough and check if everything is ready. And generate initial global 3DGS and initial poses.    
```bash
python scripts/splatam.py configs/{dataset name}/splatam.py
```  
**generate group_list first for local 3DGS generation when training**      
```bash  
python scripts/group_matrix_generate.py configs/{dataset name}/DeepGS.py
```  
**If you already have initial poses, just run script below to get rendering 3DGS**  
```bash  
python scripts/gaussian_splatting.py.py /configs/{dataset name}/post_splatam_opt.py
```  
**run DeepGS for training**  
```bash
 python scripts/splatam_test_copy.py configs/{dataset name}/DeepGS.py
```     
config `DeepGS.py` can be set like replica. Here are some attentions.  
```bash   
mode='train',   # train mode for training, eval mode for evaluation
track = True,   # set true for training
setdate_GS_per_iter = True, # will set up local 3dgs every iter
update_GS_per_iter = False, # will update global 3dgs every iter
GSupdate = False, # update whole global 3dgs with estimate poses
```   
Set train mode for training and then eval mode for evaluation.  

If you are first training(no model to load), comment out lines `load model` before running the code.  
```bash   
if (mode == 'train'):
        print('load model')
        save_dir = os.path.join('./pcd_save', config['run_name'], f"GSsetup_best_model.pth")
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
    elif (mode == 'eval'):
        print('load model')
        save_dir = os.path.join('./pcd_save', config['run_name'], f"GSsetup_best_model.pth")
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
``` 
and add them later after you train models.