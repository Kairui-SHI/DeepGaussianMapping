This is a preview version for editing ...

For details on how to obtain prepared dataset and the format of the dataset, see SplaTAM:

https://github.com/spla-tam/SplaTAM

For details on L-net or the scripts: /scripts/deepmapping.py , see DeepMapping2:

https://github.com/ai4ce/DeepMapping2


**run SplaTAM for tracking initial poses**

'''bash
python scripts/splatam.py configs/replica/splatam.py
'''


**If you already have initial poses, just run script below to get rendering 3DGS**

'''bash
python scripts/gaussian_splatting.py.py /configs/{dataset name}/post_splatam_opt.py
'''

**run DeepGS for L-net training**

'''bash
 python scripts/splatam_test_copy.py configs/replica/DeepGS.py
'''


**The code is still in the debugging phase, and there are many details such as absolute paths and mode switches that need to be adjusted!**