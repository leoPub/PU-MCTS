import copy
import itertools
import numpy as np
import pickle
import os
from search_tree import TreeNode


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class MCTS(object):

    def __init__(self, args, c_puct=5, n_rollout=10000):
        from mctsMultiVehEnv import buildEnv
        self.root_node = TreeNode(args, None, 1.0)
        self.c_puct = c_puct
        self.n_rollout = n_rollout
        self.rollout_env = buildEnv(args,
                                    render=args.render_rollout,
                                    print_warnings=False)()
        self.step_depth = 0
        self.args = args

    # def rl_acts_to_joint(self, rl_acts):

    def rollout(self, rollout_env, reward):  # 在某一中间状态下进行rollout，使用蒙特卡洛法确定动作
        node = self.root_node
        end = False
        while 1:
            if node.has_no_leaf() or end:
                if end:
                    node.update_recursive(self.rollout_env.acml_rew/self.rollout_env.steps - self.args.CTR_delt_g)
                break
            action, node = node.select(self.c_puct)  # Greedily select next move.
            reward, end, _, sum_rew, _ = rollout_env.execute_actions(action)
            reward = reward - self.args.CTR_delt_g
            self.rollout_env.steps += 1
            self.rollout_env.acml_rew += sum_rew
            self.step_depth += 1

        if not end:
            rl_ids = rollout_env.k.vehicle.get_rl_ids()
            veh_ids_main = rollout_env.k.vehicle.get_ids()
            veh_ids, veh_types, veh_lanes, x_pos, velo, veh_targ, state_dict, veh_map, left_hv, hv, right_hv \
                = rollout_env.get_vehicles_info(veh_ids_main)
            legal_acts = rollout_env.get_available_action(rl_ids, state_dict, left_hv, hv, right_hv)
            if self.args.exp_extend:
                values = [self.exp_value(rollout_env, act) - self.args.CTR_delt_g for act in legal_acts]
            else:
                values = np.zeros(len(legal_acts))

            probs = softmax(values)
            action_probs_values = zip(legal_acts, probs, values)
            node.expand(action_probs_values)

        node_value = reward

        node.update_recursive(node_value)

        # if node.parent_node and para_update:
        bad_value = node_value if self.args.rew_method in ['HDR', 'GNR'] else node_value + self.args.CTR_delt_g
        if node.parent_node and bad_value < 0 and self.args.para_update:  # TODO 该逻辑 如有问题 需要进一步优化
            for similar_action in find_group(action, rollout_env.rl_infos['step_bad']):
                for par_action, par_node in node.parent_node.children_nodes.items():
                    if similar_action == par_action:
                        par_node.update_recursive(node_value,
                                                  layer_mark=0,
                                                  par_mark=True)

    def get_action_probs(self, dir_name, reward, main_env):
        temp = 1
        rollout_pos, rollout_lane, rollout_speed, rollout_acc = main_env.get_state()
        veh_attr = main_env.cur_pos(self.args.rollout_state_path)

        rl_infos = copy.deepcopy(main_env.rl_infos)
        self.rollout_env.bypass_step()
        self.rollout_env.repos(rollout_pos, rollout_lane, rollout_speed, veh_attr, infos=rl_infos)

        rollout_log = {'legal_acts': [],
                       'n_vis': [],
                       'q_value': [],
                       'ucb_value': [],
                       'depth': [], }
        # set(main_env.legal_acts)   set(self.root_node.children_nodes.keys())
        node_to_expand = list(set(main_env.legal_acts) - set(self.root_node.children_nodes.keys()))
        node_to_minus = list(set(self.root_node.children_nodes.keys()) - set(main_env.legal_acts))
        if len(node_to_expand) > 0:
            if self.args.exp_extend:
                values = [self.exp_value(self.rollout_env, act) - self.args.CTR_delt_g for act in node_to_expand]
            else:
                values = np.zeros(len(node_to_expand))
            probs = softmax(values)
            action_probs_values = zip(node_to_expand, probs, values)
            self.root_node.expand(action_probs_values)
        for key in node_to_minus:
            self.root_node.children_nodes.pop(key)
        self.step_depth = 0
        for n in range(self.n_rollout):
            self.rollout_env.steps = main_env.steps
            self.rollout_env.acml_rew = main_env.acml_rew
            self.rollout(self.rollout_env, reward)
            self.rollout_env.repos(rollout_pos, rollout_lane, rollout_speed, veh_attr, infos=rl_infos)

            if self.args.data_save and main_env.steps in self.args.rollout_log_list:
                acts = [key for key in self.root_node.children_nodes]
                n_vis = [value.n_visits for value in self.root_node.children_nodes.values()]
                Q = [value.Q_value for value in self.root_node.children_nodes.values()]
                ucb = [value.get_ucb_value(self.c_puct) for value in self.root_node.children_nodes.values()]
                rollout_log['legal_acts'].append(acts)
                rollout_log['n_vis'].append(n_vis)
                rollout_log['q_value'].append(Q)
                rollout_log['ucb_value'].append(ucb)
                rollout_log['depth'].append(self.step_depth)

            self.step_depth = 0

        if self.args.data_save and main_env.steps in self.args.rollout_log_list:
            with open(f'{dir_name}/step{main_env.steps}_rollout_infos.pk', 'wb') as f:
                pickle.dump(rollout_log, f)

        # calc the move probabilities based on visit counts at the root node
        act_visits_values = [(act, node.n_visits, node.Q_value) for act, node in self.root_node.children_nodes.items()]
        acts, visits, values = zip(*act_visits_values)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        action = acts[np.argmax(np.array(values) + 0 * softmax(visits))]  # 可能需要调整

        return acts, act_probs, values, action

    def update_with_move(self, last_move):
        if last_move in self.root_node.children_nodes:
            self.root_node = self.root_node.children_nodes[last_move]
            self.root_node.parent_node = None
        else:
            self.root_node = TreeNode(self.args, None, 1.0)

    def __str__(self):
        return "MCTS"

    def exp_value(self, rollout_env, act):
        rl_actions = rollout_env.joint_2_dis_action(act)
        r_trd, r_arg, _ = rollout_env.compute_hdr_reward(rl_actions)
        r_hdr = self.args.w_hdr * (self.args.w_trd * r_trd + (1 - self.args.w_trd) * r_arg) / self.args.num_cav
        r_freq = 0
        lc_last_count = rollout_env.rl_infos['last_lc']
        for rl_id, action in rl_actions.items():
            # v_x = speeds[rl_id]
            if action not in [3, 4, 5]:
                r_freq -= np.exp(-lc_last_count[rl_id])
        EE_val = r_hdr + self.args.w_trd * r_freq / self.args.num_cav
        return EE_val * sum(self.args.gamma ** i for i in range(0, self.args.sims_per_step))


def find_group(action, bad_dict):
    rl_ids = [veh_id for veh_id in bad_dict]
    ACTION_GROUPS = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]
    FULL_ACTION_SET = list(range(9))
    NUM_AGENTS = len(rl_ids)
    total_contaminated_actions = set()
    for i, agent_id in enumerate(rl_ids):
        is_bad = bad_dict.get(agent_id) == 1
        if is_bad:
            source_action = action[i]
            group_index = source_action // 3
            contamination_group = ACTION_GROUPS[group_index]
            action_sets_for_this_source = [FULL_ACTION_SET] * NUM_AGENTS
            action_sets_for_this_source[i] = contamination_group
            contaminated_subset = itertools.product(*action_sets_for_this_source)
            total_contaminated_actions.update(contaminated_subset)
    return list(total_contaminated_actions)


