import numpy as np
import math


class TreeNode(object):
    def __init__(self, args, parent, prior_p):
        self.parent_node = parent
        self.children_nodes = {}  # a map from action to TreeNode
        self.n_visits = 0
        self.Q_value = 0
        self.u_value = 0
        self.R_value = 0
        self.re_value = 0
        self.p_value = prior_p
        self.args = args

    def expand(self, action_priors_value):
        for action, prob, value in action_priors_value:  # 使用list(action_priors)查看详细信息
            if action not in self.children_nodes:
                self.children_nodes[action] = TreeNode(self.args, self, prob)
                self.children_nodes[action].Q_value = value

    def select(self, c_puct):
        return max(self.children_nodes.items(),
                   key=lambda act_node: act_node[1].get_ucb_value(c_puct))

    def update(self, leaf_value, layer_mark, par_mark):

        # if leaf_value > 1:
        #     layer_mark = 0

        if par_mark:
            decay_ef = self.args.par_decay ** layer_mark
        else:
            self.n_visits += 1
            decay_ef = self.args.update_decay ** layer_mark

        self.re_value += decay_ef
        self.R_value += leaf_value * decay_ef
        self.Q_value = self.R_value / self.re_value

    def update_recursive(self, leaf_value, layer_mark=0, par_mark=False):
        # If it is not root, this node's parent should be updated first.
        # but if not using n_visit, the
        self.update(leaf_value, layer_mark, par_mark)
        if self.parent_node:
            layer_mark += 1
            self.parent_node.update_recursive(leaf_value, layer_mark, par_mark)

    def get_ucb_value(self, c_puct):
        self.u_value = (c_puct * self.p_value * check_sign(self.Q_value) *
                        np.sqrt(math.log(max(1, self.parent_node.n_visits)) / (1 + self.n_visits)))
        return self.Q_value + self.u_value

    def has_no_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self.children_nodes == {}

    def is_root(self):
        return self.parent_node is None

    def show_nodes(self, c_puct):
        acts = [key for key in self.children_nodes]
        n_vis = [value.n_visits for value in self.children_nodes.values()]
        Q = [value.Q_value for value in self.children_nodes.values()]
        ucb = [value.get_ucb_value(c_puct) for value in self.children_nodes.values()]
        return np.column_stack((acts, n_vis, Q, ucb))


# 这个函数忘了，应该需要调整
def check_sign(value):
    return 1
    # if value >= 0:
    #     return 1
    # else:
    #     return -1
