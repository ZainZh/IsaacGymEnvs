## experiment date: 2022.06.29
    task = DualFranka (whole task)

    algo: CQL+ one step replaygbuffer data

    the training epoch: 1800

    the training rewards: 45000

    result: fail
	
## experiment date: 2022.07.02 
    
    task = DualFranka (task1 spoon)

    algo: PPO with non-conversetive params (gamma: 0.9, horizon_length: 2000, minibatch_size: 16384)

    the training epoch: 130

    the training rewards: 55000

    result: success

## experiment date: 2022.07.02
    
    task = DualFranka (task1 cup)

    algo: PPO with non-conversetive params (gamma: 0.9, horizon_length: 2000, minibatch_size: 16384)

    the training epoch: 150

    the training rewards: 25000

    result: success


## experiment date: 2022.07.02
    
    task = DualFranka (task1 cup and spoon)
    
    task_name = DualFrankaPPO_0702_stage1 (task1 cup and spoon)

    algo: PPO with non-conversetive params (gamma: 0.9, horizon_length: 2000, minibatch_size: 16384)

    the training epoch: 350

    the training rewards: 60000

    result: spoon successes but cup fail. rewards are stuck in the locally optimal solution

## experiment date: 2022.07.02
    
    task = DualFranka (task1 cup and spoon)
    
    task_name = DualFrankaPPO_0702_stage1_3 (task1 cup and spoon)

    algo: PPO (normalize_value: True, entropy_coef: 0.01, horizon_length: 2000, minibatch_size: 16384 )

    the training epoch: 150

    the training rewards: 15000

    result: spoon successes but cup fails. rewards are stuck in the locally optimal solution

## experiment date: 2022.07.02
    
    task = DualFranka (task1 cup and spoon)
    
    task_name = DualFrankaPPO_0702_stage1_2 (task1 cup and spoon)

    algo: PPO (normalize_value: True, entropy_coef: 0.01,  e_clip: 0.2, horizon_length: 4000, minibatch_size: 32768 )

    the training epoch: 150

    the training rewards: 15000

    result: spoon successes but cup fails. rewards are stuck in the locally optimal solution

## experiment date: 2022.07.02
    
    task = DualFranka (task1 cup and spoon)
    
    task_name = DualFrankaPPO_0702_stage1_nolift (task1 cup and spoon)

    algo: PPO (normalize_value: True, entropy_coef: 0.01,  e_clip: 0.2, horizon_length: 4000, minibatch_size: 32768 )

    the training epoch: 150

    the training rewards: 15000

    result: spoon and cup both success, but cup was always knocked off.

## experiment date: 2022.07.03
    
    task = DualFranka (task1 cup and spoon)
    
    task_name = DualFrankaPPO_0702_stage1_purestage1

    algo: PPO (without any changed  parames)

    the training epoch: 350

    the training rewards: 30000(very strange!!!!  Because in the test.py, each env only can obtain maxinum 16 rewards.)

    result: spoon and cup both are very close to success, but then perform poor!