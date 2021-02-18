import os

def run_exp(gpu,imb_type,imb_factor,loss_type,train_rule,exp_str,normalize_type):
    # python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW
    the_command = "python cifar_train.py " \
        + " --gpu="+str(gpu) \
        + " --imb_type="+imb_type \
        + " --imb_factor="+str(imb_factor) \
        + " --loss_type="+loss_type \
        + " --train_rule="+train_rule \
        + " --exp_str="+exp_str \
        + " --normalize_type="+normalize_type 
    
    print(the_command)
    os.system(the_command)


# run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="logit_normalization",exp_str='ln_1')
# run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="prob_normalization",exp_str='pn_1')

# run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="logit_normalization",exp_str='ln_2')
# run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="logit_normalization",exp_str='ln_3')

# run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="prob_normalization",exp_str='pn_1')
# run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="prob_normalization",exp_str='pn_2')
# run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="prob_normalization",exp_str='pn_3')


run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="logit_standardization",exp_str='ls_1')
run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="prob_standardization",exp_str='ps_1')

run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="logit_standardization",exp_str='ls_2')
run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="logit_standardization",exp_str='ls_3')

run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="prob_standardization",exp_str='ps_2')
run_exp(gpu=0,imb_type='exp',imb_factor=0.02,loss_type="LDAM",train_rule="DRW",normalize_type="prob_standardization",exp_str='ps_3')