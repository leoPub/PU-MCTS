from __future__ import print_function
import os
import shutil

from run_config import run_config
from types import SimpleNamespace as simNp
from config import config
import pickle
from mcts_veh import MCTSVeh
from mctsMultiVehEnv import buildEnv
import uuid

args = simNp(**config)

args.data_save = False
# args.data_save = True

# args.main_render = False
args.main_render = True
args.render_rollout = False
# args.render_rollout = True
config_dict = {1: 'SN', 2: 'PN', 3: 'SE', 4: 'PE', 5: 'SER', 6: 'PER'}
args.sims_per_step = 1

args.num_cav = 2
args.num_hdv = 6
args.update_decay = pow(args.gamma, args.sims_per_step)
args.par_decay = 0.1

args.rew_method = 'HDR'  # HDR CTR GNR CTH
config_num = 1
args.para_update, args.pre_extend, args.exp_extend = run_config(config_num)

# c_puct_list = [10, 20, 30, 40, 50, 60, 70, 90, 120, 150, 1000, 2000]
# n_rollout_list = [200, 400, 500, 600, 700]
n_rollout_list = [200]
c_puct_list = [120]

exp_num = 200
# EXP_LIST = [2]

args.rollout_log_list = [0]


with open(f'mcts_utils/{args.num_cav}cav{args.num_hdv}hdv_initial_configs.pkl', 'rb') as file:  # 注意是 'rb' 模式（二进制读取）
    initial_configs = pickle.load(file)
with open(f'mcts_utils/{2}cav{6}hdv_initial_configs.pkl', 'rb') as file:  # 注意是 'rb' 模式（二进制读取）
    initial_configs_6 = pickle.load(file)
args.initial_config = initial_configs[0]  # 初始化为一个临时的场景
n_rollout_list = [n_rollout_list] if isinstance(n_rollout_list, int) else n_rollout_list
c_puct_list = [c_puct_list] if not isinstance(c_puct_list, list) else c_puct_list

unique_filename = f"../tb_logs/tmp_xmls/state_{uuid.uuid4()}.xml"
# with open(unique_filename, 'w') as f:
#     pass
args.rollout_state_path = os.path.abspath(unique_filename)

for c_puct in c_puct_list:
    for n_rollout in n_rollout_list:

        dir_name_base = (f'/XXX/'
                         f'hdv{args.num_hdv}/{config_dict[config_num]}')

        mcts_veh = MCTSVeh(args,
                           c_puct=c_puct,
                           n_rollout=n_rollout,
                           is_selfplay=1)

        for k in range(xxxx):

            dir_name = f"{dir_name_base}/exp_{k}"

            if args.data_save:
                if os.path.exists(dir_name):
                    continue
                os.makedirs(dir_name, exist_ok=True)

            folder_path = './saved_pics'
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                try:
                    # 情况 A: 如果是文件 或 符号链接
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # unlink 等同于 remove，用于删除文件
                        print(f"已删除文件: {filename}")

                    # 情况 B: 如果是目录
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # rmtree 用于递归删除文件夹及其内容
                        print(f"已删除文件夹: {filename}")

                except Exception as e:
                    print(f"删除 {file_path} 失败. 原因: {e}")


            args.initial_config = initial_configs[k]
            mcts_veh.mcts.args = args

            # 创建主环境，并进入初始状态
            main_env = buildEnv(args,
                                render=args.main_render,
                                print_warnings=False)()
            main_env.bypass_step()  # 主环境进入初始状态

            rl_ids_main = main_env.k.vehicle.get_rl_ids()
            veh_ids_main = main_env.k.vehicle.get_ids()
            veh_ids, veh_types, veh_lanes, x_pos, velo, veh_targ, state_dict, veh_map, left_hv, hv, right_hv \
                = main_env.get_vehicles_info(veh_ids_main)
            time_since_last_lc = {veh_id: args.t_lc_freq + 1 for veh_id in rl_ids_main}
            veh_ttc = {veh: main_env.get_ttc(veh, state_dict, hv) for veh in veh_ids_main}
            main_env.rl_infos['ttc'] = veh_ttc
            main_env.rl_infos['last_lc'] = time_since_last_lc
            main_env.rl_infos['arrived_status'] = {veh: 0 for veh in rl_ids_main}
            main_env.rl_infos['step_bad'] = {veh: 0 for veh in rl_ids_main}
            main_env.steps = 0
            main_env.acml_rew = 0
            reward = 0
            sim_step = 0
            log_info = {}

            while True:

                main_env.save_custom_screenshot(f'./saved_pics/step_{sim_step}.png')
                main_env.legal_acts = main_env.get_available_action(rl_ids_main, state_dict, left_hv, hv, right_hv)
                # if args.num_cav != 0:
                action, action_probs, acts, q_values = mcts_veh.get_action(main_env, dir_name, reward)
   
                reward, end, _, sum_reward, _ = main_env.execute_actions(action)
                reward = reward - args.CTR_delt_g
                log_info.update({sim_step: {
                    'veh_ids': veh_ids,
                    'x_pos': x_pos,
                    'veh_lanes': veh_lanes,
                    'velo': velo,
                    'target': veh_targ,
                    'veh_types': veh_types,
                    'state_dict': state_dict,
                    'veh_map': veh_map,
                    'left_hv': left_hv,
                    'hv': hv,
                    'right_hv': right_hv,
                    'action': action,
                    'legal_acts': acts,
                    'q_values': q_values,  # 多步合并是否会带来Q值分布不一致的情况？
                    'reward_acml': sum_reward,
                    'ttc': main_env.rl_infos['ttc'],
                    'last_lc': main_env.rl_infos['last_lc'],
                }})
                main_env.steps += 1
                main_env.acml_rew += sum_reward

                veh_ids_main = main_env.env_in_main_road()
                veh_ids, veh_types, veh_lanes, x_pos, velo, veh_targ, state_dict, veh_map, left_hv, hv, right_hv \
                    = main_env.get_vehicles_info(veh_ids_main)

                sim_step += 1

                if end:
                    if args.data_save:
                        log_info[sim_step - 1].update({'arrive_status': main_env.rl_infos['arrived_status']})
                        with open(f'{dir_name}/simu_traj.pkl', 'wb') as file:  # 注意使用二进制写入模式 'wb'
                            pickle.dump(log_info, file)
                    main_env.terminate()
                    mcts_veh.reset_mcts_veh()
                    break

        mcts_veh.mcts.rollout_env.terminate()
print('ALL FINISHED')
