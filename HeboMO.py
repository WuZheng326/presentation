import hebo
import torch
import numpy as np
import pandas as pd

from hebo.optimizers.general import GeneralBO
from hebo.design_space.design_space import DesignSpace

from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment

from botorch.utils.multi_objective.pareto import is_non_dominated

class MultiTask_BO(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space):
        AbstractOptimizer.__init__(self, design_space)

        self.n_suggestions = 4
        self.initial_sample_size = 10
        self.suggest_corner = True
        self.initial_sample = True
        self.first_suggestion = True

        self.vec_set = self.construct_microarchitecture_vec_set()
        self.embedding_set = self.construct_microarchitecture_embedding_set()

        self.dim, self.bounds, self.valid_dim = self.construct_bounds()

        vec_set = np.array(self.vec_set)
        max_value = np.max(vec_set, axis=0)
        min_value = np.min(vec_set, axis=0)

        params =[
            {'name': 'Fetch', 'type': 'int', 'lb': min_value[0], 'ub': max_value[0]},
            {'name': 'Decoder', 'type': 'int', 'lb': min_value[1], 'ub': max_value[1]},
            {'name': 'ISU', 'type': 'int', 'lb': min_value[2], 'ub': max_value[2]},
            {'name': 'IFU', 'type': 'int', 'lb': min_value[3], 'ub': max_value[3]},
            {'name': 'ROB', 'type': 'int', 'lb': min_value[4], 'ub': max_value[4]},
            {'name': 'PRF', 'type': 'int', 'lb': min_value[5], 'ub': max_value[5]},
            {'name': 'LSU', 'type': 'int', 'lb': min_value[6], 'ub': max_value[6]},
            {'name': 'I-Cache/MMU', 'type': 'int', 'lb': min_value[7], 'ub': max_value[7]},
            {'name': 'D-Cache/MMU', 'type': 'int', 'lb': min_value[8], 'ub': max_value[8]},
        ]

        self.space = DesignSpace().parse(params)

        self.bo = GeneralBO(self.space, num_obj=3, rand_sample=10)

    def construct_microarchitecture_vec_set(self):
        microarchitecture_vec_set = []
        for i in range(1, self.design_space.size + 1):
            microarchitecture_vec_set.append(
                self.design_space.idx_to_vec(i)#id转换成向量，各个模块里的设计参数组
            )
        # np.save('micro.npy', np.array(microarchitecture_vec_set))
        return microarchitecture_vec_set

    def construct_microarchitecture_embedding_set(self):
        microarchitecture_embedding_set = []
        for i in range(1, self.design_space.size + 1):
            microarchitecture_embedding_set.append(
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(i)#id转换成向量，各个模块里的设计参数组
                )
            )
        np.save('micro.npy', np.array(microarchitecture_embedding_set))
        return torch.Tensor(microarchitecture_embedding_set)

    def microarchitecture_embedding_to_vec(self, embedding):
        acc_l = 0
        component_dims = self.design_space.component_dims[0]
        vec = []
        for i, c in enumerate(self.design_space.components):
            l = len(self.design_space.components_mappings[c]["description"])
            component_val = embedding[acc_l:acc_l + l]
            acc_l += l
            # if i in self.valid_dim:
            for j in range(component_dims[i]):
                if check_equal_list(component_val, self.design_space.components_mappings[c][j + 1]):
                    vec.append(j + 1)
                    break
        return vec

    def construct_bounds(self):
        component_dims = self.design_space.component_dims[0]
        valid_dim = np.where(np.array(component_dims) > 1)[0]
        # print('valid dim:', valid_dim)
        dim = len(valid_dim)
        bounds = torch.ones(2, dim)
        bounds[1] = torch.from_numpy(np.array(component_dims)[valid_dim].flatten())
        # print('bounds:', bounds)
        return dim, bounds, valid_dim

    def suggest(self):
        if self.first_suggestion==True:

            #corner case
            is_max_corner = is_non_dominated(self.embedding_set)
            is_min_corner = is_non_dominated(self.embedding_set * (-1))
            corner_case = []
            for i in range(self.design_space.size):
                if is_max_corner[i] or is_min_corner[i]:
                    corner_case.append(self.embedding_set[i].type(torch.int).numpy().tolist())
            print('corner case:', corner_case)

            #random sample
            samp = self.space.sample(num_samples=self.initial_sample_size)
            samp = np.array(samp)
            samp = samp.tolist()

            embedding = [
                self.design_space.vec_to_microarchitecture_embedding(
                    _samp
                ) for _samp in samp
            ]

            self.first_suggestion = False

            return corner_case + embedding

        else:
            samp = self.bo.suggest(n_suggestions=self.n_suggestions)
            samp = np.array(samp)
            samp = samp.tolist()

            embedding = [
                self.design_space.vec_to_microarchitecture_embedding(
                    _samp
                ) for _samp in samp
            ]

            return embedding

    def observe(self, x, y):
        x_vec = [self.microarchitecture_embedding_to_vec(_x) for _x in x]
        pD_x = pd.DataFrame(x_vec)
        pD_x.columns = ['Fetch', 'Decoder', 'ISU', 'IFU', 'ROB', 'PRF', 'LSU', 'I-Cache/MMU', 'D-Cache/MMU']

        ob_y  = np.array(y)
        self.bo.observe(pD_x, ob_y)


def check_equal_list(list1, list2):
    assert len(list1) == len(list2), assert_error("unequal list length")
    return sum([_list1 == _list2 for (_list1, _list2) in zip(list1, list2)]) == len(list1)

if __name__ == "__main__":
    experiment(MultiTask_BO)
