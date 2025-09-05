rm *.zip
###
 # @Author       : ZhangYang zzzy0223@qq.com
 # @LastEditTime : 2025-09-05 16:14
 # @FilePath     : /alqq_genrec/env.sh
 # 
 # coded by ZhangYang@BUPT, my email is zhangynag0706@bupt.edu.cn
### 
export TRAIN_DATA_PATH=/mnt/hdd/zhangyang/alqq_genrec_old/TencentGR_1k
CUR=$(pwd)
export TRAIN_TF_EVENTS_PATH=$CUR/working_dir/tensorboard
export TRAIN_LOG_PATH=$CUR/working_dir/log
export TRAIN_CKPT_PATH=$CUR/working_dir/ckpt
export RUNTIME_SCRIPT_DIR=$CUR
export USER_CACHE_PATH=/mnt/hdd/zhangyang/alqq_genrec/USARCACHE